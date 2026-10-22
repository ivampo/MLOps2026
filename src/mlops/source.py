import argparse
import asyncio
import csv
import logging
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import cycle, islice
from pathlib import Path
from uuid import UUID, uuid4, uuid5

from mlops.config import get_config
from mlops.logging import setup_logging
from mlops.messaging import create_broker, declare_queues, publish_with_retry
from mlops.schemas import IrisFeatures, TaskMessage

logger = logging.getLogger(__name__)


def read_samples(path: Path | None, count: int) -> Iterator[IrisFeatures]:
    if path is not None:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in islice(csv.DictReader(stream), count):
                yield IrisFeatures.model_validate({key: float(value) for key, value in row.items()})
    else:
        from mlops.data import FEATURES, load_dataset

        rows = load_dataset().loc[:, FEATURES].to_dict(orient="records")
        for row in islice(cycle(rows), count):
            yield IrisFeatures.model_validate(row)


async def produce(path: Path | None, count: int, batch_id: UUID) -> None:
    config = get_config()
    started = time.perf_counter()
    sent = 0
    async with create_broker(config) as broker:
        await declare_queues(broker, config)
        for index, sample in enumerate(read_samples(path, count)):
            task = TaskMessage(
                task_id=uuid5(batch_id, str(index)),
                submitted_at=datetime.now(UTC),
                samples=[sample],
            )
            await publish_with_retry(
                broker,
                config,
                task.model_dump(mode="json"),
                queue=config.rabbitmq_queue,
                message_id=str(task.task_id),
            )
            sent += 1
            logger.info(
                "task published",
                extra={
                    "fields": {
                        "batch_id": str(batch_id),
                        "sent": sent,
                        "elapsed_s": round(time.perf_counter() - started, 3),
                    }
                },
            )
    logger.info("source finished", extra={"fields": {"sent": sent, "batch_id": str(batch_id)}})


def main() -> None:
    parser = argparse.ArgumentParser(description="Пакетный источник задач с backpressure")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-id", type=UUID, default=None)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    setup_logging(get_config().log_level)
    asyncio.run(produce(args.csv, args.count, args.batch_id or uuid4()))


if __name__ == "__main__":
    main()
