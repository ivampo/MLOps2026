import asyncio
import logging

from aio_pika.exceptions import AMQPException
from faststream.rabbit import Channel, RabbitBroker, RabbitQueue
from faststream.rabbit.schemas.queue import ClassicQueueArgs
from faststream.security import SASLPlaintext

from mlops.config import Config

logger = logging.getLogger(__name__)
PUBLISH_ERRORS = (AMQPException, OSError, TimeoutError)


def create_broker(config: Config, *, prefetch_count: int | None = None) -> RabbitBroker:
    return RabbitBroker(
        host=config.rabbitmq_host,
        port=config.rabbitmq_port,
        security=SASLPlaintext(
            username=config.rabbitmq_user,
            password=config.rabbitmq_password.get_secret_value(),
        ),
        default_channel=Channel(
            prefetch_count=prefetch_count,
            publisher_confirms=True,
            on_return_raises=True,
        ),
        graceful_timeout=60.0,
    )


def get_task_queue(config: Config) -> RabbitQueue:
    arguments: ClassicQueueArgs = {
        "x-dead-letter-exchange": "",
        "x-dead-letter-routing-key": f"{config.rabbitmq_queue}.dlq",
    }
    if config.backpressure_enabled:
        arguments.update(
            {"x-max-length": config.rabbitmq_queue_max_length, "x-overflow": "reject-publish"}
        )
    return RabbitQueue(config.rabbitmq_queue, durable=True, arguments=arguments)


async def declare_queues(broker: RabbitBroker, config: Config) -> None:
    await broker.declare_queue(RabbitQueue(f"{config.rabbitmq_queue}.dlq", durable=True))
    await broker.declare_queue(get_task_queue(config))


async def publish_with_retry(
    broker: RabbitBroker,
    config: Config,
    body: dict | bytes,
    *,
    queue: str,
    message_id: str,
    headers: dict | None = None,
) -> None:
    delay = config.retry_delay
    while True:
        try:
            await broker.publish(
                body,
                queue=queue,
                persist=True,
                mandatory=True,
                message_id=message_id,
                headers=headers,
                timeout=config.publish_timeout,
            )
            return
        except PUBLISH_ERRORS:
            logger.debug("publish deferred", extra={"fields": {"queue": queue}})
            await asyncio.sleep(delay)
            delay = min(delay * 2, 0.5)
