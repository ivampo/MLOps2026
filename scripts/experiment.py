import argparse
import csv
import json
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4, uuid5

import docker
import httpx

from mlops.config import get_config

ROOT = Path(__file__).resolve().parents[1]


def compose(*args, override=None):
    command = ["docker", "compose", "-f", str(ROOT / "docker-compose.yml")]
    if override is not None:
        command += ["-f", str(override)]
    result = subprocess.run(
        command + list(args), cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE
    )
    return result.stdout.strip()


def database_stats(task_ids):
    ids = ",".join(f"'{task_id}'" for task_id in task_ids)
    sql = (
        "SELECT json_build_object('completed',count(*),'mean_ms',avg((result->>'total_ms')::float),"
        "'p95_ms',percentile_cont(0.95) WITHIN GROUP (ORDER BY (result->>'total_ms')::float),"
        "'durations_ms',json_agg((result->>'total_ms')::float ORDER BY completed_at)) "
        f"FROM task_results WHERE task_id IN ({ids});"
    )
    return json.loads(
        compose(
            "exec",
            "-T",
            "db",
            "sh",
            "-c",
            'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "$1"',
            "sh",
            sql,
        )
    )


def memory_mib(container):
    container.reload()
    if container.status != "running":
        return None
    stats = container.stats(stream=False, one_shot=True)["memory_stats"]
    inactive = stats.get("stats", {}).get("inactive_file", 0)
    return round((stats.get("usage", 0) - inactive) / 1024**2, 2)


def source_events(container):
    events = []
    for line in container.logs().decode().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("message") == "task published":
            events.append(item)
    return events


def run_scenario(name, workers, bounded, count, delay, output):
    config = get_config()
    batch_id = uuid4()
    queue = f"experiment.{name}.{batch_id.hex[:8]}"
    env = {
        "RABBITMQ_QUEUE": queue,
        "BACKPRESSURE_ENABLED": str(bounded).lower(),
        "WORKER_CONCURRENCY": "1",
        "WORKER_PREFETCH_COUNT": "1",
        "INFERENCE_DELAY": str(delay),
        "RABBITMQ_QUEUE_MAX_LENGTH": "10",
    }
    client = docker.from_env()
    broker_container = client.containers.get(compose("ps", "-q", "rabbitmq"))
    source = None
    snapshots = []
    task_ids = [uuid5(batch_id, str(index)) for index in range(count)]
    with tempfile.TemporaryDirectory(prefix="mlops-experiment-") as temp:
        override = Path(temp) / "compose.json"
        override.write_text(
            json.dumps(
                {
                    "services": {
                        "worker": {"environment": env},
                        "source": {"environment": env},
                    }
                }
            )
        )
        compose("stop", "worker")
        compose(
            "up",
            "-d",
            "--no-deps",
            "--wait",
            "--scale",
            f"worker={workers}",
            "worker",
            override=override,
        )
        worker_containers = [
            client.containers.get(cid)
            for cid in compose("ps", "-q", "worker", override=override).splitlines()
        ]
        try:
            cid = compose(
                "run",
                "-d",
                "--no-deps",
                "--name",
                f"mlops-source-{batch_id.hex[:8]}",
                "source",
                "mlops-source",
                "--count",
                str(count),
                "--batch-id",
                str(batch_id),
                override=override,
            )
            source = client.containers.get(cid)
            started = time.monotonic()
            previous_sent = 0
            previous_time = started
            with httpx.Client(
                auth=(config.rabbitmq_user, config.rabbitmq_password.get_secret_value()),
                base_url="http://127.0.0.1:15672",
                timeout=5,
            ) as management:
                while time.monotonic() - started < 300:
                    now = time.monotonic()
                    queue_info = (
                        management.get(f"/api/queues/%2F/{queue}").raise_for_status().json()
                    )
                    events = source_events(source)
                    sent = len(events)
                    stats = database_stats(task_ids)
                    snapshot = {
                        "seconds": round(now - started, 3),
                        "ready": queue_info.get("messages_ready", 0),
                        "unacked": queue_info.get("messages_unacknowledged", 0),
                        "source_rate": round(
                            (sent - previous_sent) / max(now - previous_time, 0.001), 2
                        ),
                        "sent": sent,
                        "completed": stats["completed"],
                        "source_mib": memory_mib(source),
                        "worker_mib": round(sum(memory_mib(c) or 0 for c in worker_containers), 2),
                        "broker_mib": memory_mib(broker_container),
                    }
                    snapshots.append(snapshot)
                    previous_sent, previous_time = sent, now
                    source.reload()
                    if source.status == "exited" and source.attrs["State"]["ExitCode"] != 0:
                        raise RuntimeError(source.logs().decode())
                    if (
                        stats["completed"] == count
                        and source.status == "exited"
                        and snapshot["ready"] == 0
                        and snapshot["unacked"] == 0
                    ):
                        break
                    time.sleep(1)
                else:
                    raise TimeoutError(f"Scenario {name} did not drain")
            events = source_events(source)
            middle = events[count // 5 : count * 4 // 5]
            source_rate = (len(middle) - 1) / (middle[-1]["elapsed_s"] - middle[0]["elapsed_s"])
            summary = {
                "name": name,
                "workers": workers,
                "bounded": bounded,
                "tasks": count,
                "delay_s": delay,
                "queue": queue,
                "batch_id": str(batch_id),
                "max_ready": max(s["ready"] for s in snapshots),
                "max_unacked": max(s["unacked"] for s in snapshots),
                "source_rate_middle": round(source_rate, 2),
                "source_peak_mib": max(s["source_mib"] or 0 for s in snapshots),
                "worker_peak_mib": max(s["worker_mib"] for s in snapshots),
                **stats,
            }
            assert len(events) == count
            assert stats["completed"] == count
            if bounded:
                assert summary["max_ready"] <= 10
                assert summary["max_unacked"] <= workers
            window = [s for s in snapshots if count * 0.2 <= s["completed"] <= count * 0.8]
            if len(window) >= 2:
                seconds = window[-1]["seconds"] - window[0]["seconds"]
                summary["worker_rate_steady"] = round(
                    (window[-1]["completed"] - window[0]["completed"]) / seconds, 2
                )
                summary["source_rate_same_window"] = round(
                    (window[-1]["sent"] - window[0]["sent"]) / seconds, 2
                )
            with (output / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(snapshots[0]))
                writer.writeheader()
                writer.writerows(snapshots)
            print(f"{name}: {stats['completed']}/{count} tasks completed", flush=True)
            return summary
        finally:
            compose("stop", "worker", override=override)
            if source is not None:
                source.remove(force=True)
            client.close()


def plot(output, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for summary in summaries:
        name = summary["name"]
        with (output / f"{name}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        seconds = [float(row["seconds"]) for row in rows]
        axes[0, 0].plot(
            seconds, [int(row["ready"]) + int(row["unacked"]) for row in rows], label=name
        )
        axes[0, 1].plot(seconds, [float(row["source_rate"]) for row in rows], label=name)
        axes[1, 0].plot(
            seconds,
            [float(row["source_mib"]) if row["source_mib"] else float("nan") for row in rows],
            label=name,
        )
        durations = [ms / 1000 for ms in summary["durations_ms"]]
        axes[1, 1].plot(range(1, len(durations) + 1), durations, label=name)
    labels = [
        ("Pending messages (ready + unacked)", "messages"),
        ("Source accepted rate", "tasks/s"),
        ("Source memory", "MiB"),
        ("Task end-to-end latency", "seconds"),
    ]
    for ax, (title, unit) in zip(axes.flat, labels, strict=True):
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xlabel("task completion index" if ax is axes[1, 1] else "seconds")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for summary in summaries:
        with (output / f"{summary['name']}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        seconds = [float(row["seconds"]) for row in rows]
        for ax, key in zip(axes, ["worker_mib", "broker_mib"], strict=True):
            ax.plot(seconds, [float(row[key]) for row in rows], label=summary["name"])
    for ax, title in zip(axes, ["All workers memory", "RabbitMQ memory"], strict=True):
        ax.set_title(title)
        ax.set_xlabel("seconds")
        ax.set_ylabel("MiB (Docker working set)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output / "memory.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    bounded = [item for item in summaries if item["bounded"]]
    for ax, summary in zip(axes, bounded, strict=True):
        with (output / f"{summary['name']}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        seconds = [float(row["seconds"]) for row in rows]
        ax.step(seconds, [int(row["ready"]) for row in rows], where="post", label="Ready")
        ax.step(seconds, [int(row["unacked"]) for row in rows], where="post", label="Unacked")
        ax.axhline(10, color="gray", linestyle=":", label="Ready limit = 10")
        ax.set_title(summary["name"])
        ax.set_xlabel("seconds")
        ax.set_ylabel("messages")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output / "ready-unacked.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=180)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=ROOT / "results/experiment")
    args = parser.parse_args()
    if args.count < 20 or args.delay <= 0:
        parser.error("count >= 20, delay > 0 required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    try:
        for name, workers, bounded in [
            ("unlimited", 1, False),
            ("bounded-one", 1, True),
            ("bounded-two", 2, True),
        ]:
            summaries.append(run_scenario(name, workers, bounded, args.count, args.delay, output))
        plot(output, summaries)
        (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    finally:
        compose("up", "-d", "--no-deps", "--wait", "--scale", "worker=1", "worker")


if __name__ == "__main__":
    main()
