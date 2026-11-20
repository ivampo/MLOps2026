import logging
import time

from faststream.rabbit import RabbitBroker, RabbitQueue
from opentelemetry import propagate, trace

from mlops.config import Config
from mlops.telemetry import tracer

logger = logging.getLogger(__name__)


def queues(config: Config) -> tuple[RabbitQueue, RabbitQueue]:
    dead = RabbitQueue(
        f"{config.queue_name}.dlq", durable=True, arguments={"x-queue-type": "quorum"}
    )
    work = RabbitQueue(
        config.queue_name,
        durable=True,
        arguments={
            "x-queue-type": "quorum",
            "x-max-length": config.queue_limit,
            "x-overflow": "reject-publish",
            "x-delivery-limit": config.max_attempts,
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": dead.name,
            "x-dead-letter-strategy": "at-least-once",
        },
    )
    return work, dead


def create_broker(config: Config) -> RabbitBroker:
    return RabbitBroker(
        config.rabbitmq_url.get_secret_value(),
        max_consumers=1,
        graceful_timeout=60,
        logger=logger,
        # Validation belongs in the handler, including for malformed JSON messages.
        decoder=lambda message: message.body,
    )


async def declare_queues(broker: RabbitBroker, config: Config) -> None:
    for queue in reversed(queues(config)):
        await broker.declare_queue(queue)


async def publish_task(broker: RabbitBroker, config: Config, body, identifier: str) -> None:
    with tracer.start_as_current_span("publish", kind=trace.SpanKind.PRODUCER):
        headers = {"task_id": identifier, "published_at_ns": time.time_ns()}
        propagate.inject(headers)
        await broker.publish(
            body,
            queue=config.queue_name,
            correlation_id=identifier,
            headers=headers,
            persist=True,
            timeout=config.publish_timeout,
        )
        logger.info("task published")
