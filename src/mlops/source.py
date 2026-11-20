import argparse
import asyncio
import logging
import signal
import time
from uuid import uuid4

from aiormq.exceptions import DeliveryError

from mlops.broker import create_broker, declare_queues, publish_task
from mlops.config import get_config
from mlops.logging import setup_logging
from mlops.telemetry import instruments, setup_telemetry, shutdown_telemetry, task_context, tracer

logger = logging.getLogger(__name__)


async def send(broker, config, body: str) -> None:
    identifier = str(uuid4())
    with task_context(identifier), tracer.start_as_current_span("source.task") as span:
        while True:
            try:
                # Send raw bytes so the poison-message scenario also reaches the worker.
                await publish_task(broker, config, body.encode(), identifier)
                instruments.published.add(1)
                return
            except DeliveryError:
                instruments.rejected.add(1)
                logger.warning("broker rejected publish; source waiting")
                span.add_event("broker.nack")
                started = time.perf_counter()
                with tracer.start_as_current_span("source.wait"):
                    await asyncio.sleep(config.source_retry_delay)
                instruments.wait.add(time.perf_counter() - started)


async def run(path: str) -> None:
    config = get_config()
    setup_logging(config.log_level)
    setup_telemetry("source", config.inference_delay)
    broker = create_broker(config)
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    if hasattr(loop, "add_signal_handler"):
        try:
            loop.add_signal_handler(signal.SIGTERM, task.cancel)
        except NotImplementedError:
            pass  # Windows has no loop signal handlers; containers use Linux.
    try:
        async with broker:
            await declare_queues(broker, config)
            with open(path, encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        await send(broker, config, line)
        logger.info("source completed")
    finally:
        logger.info("graceful shutdown started")
        await asyncio.to_thread(shutdown_telemetry)
        logger.info("graceful shutdown completed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a JSONL file with bounded memory")
    parser.add_argument("path")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.path))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception as exc:
        logger.error("source failed", extra={"fields": {"error_type": type(exc).__name__}})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
