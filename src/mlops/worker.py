import asyncio
import logging
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

from faststream import Context, ContextRepo
from faststream.asgi import AsgiFastStream, make_ping_asgi
from faststream.middlewares import AckPolicy
from faststream.rabbit import RabbitMessage
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from mlops.config import Config, get_config
from mlops.db import get_engine
from mlops.logging import setup_logging
from mlops.messaging import create_broker, declare_queues, get_task_queue, publish_with_retry
from mlops.model import LoadedModel, load_model
from mlops.schemas import TaskMessage, TaskResult
from mlops.storage import get_result, init_storage, next_attempt, save_result

logger = logging.getLogger(__name__)


def predict(task: TaskMessage, model: LoadedModel, delay: float) -> TaskResult:
    started = time.perf_counter()
    time.sleep(delay)
    predictions = model.predict(task.samples)
    return TaskResult(
        model=model.info,
        predictions=predictions,
        processing_ms=round((time.perf_counter() - started) * 1000, 2),
        total_ms=round((datetime.now(UTC) - task.submitted_at).total_seconds() * 1000, 2),
    )


def create_worker(config: Config) -> AsgiFastStream:
    prefetch = config.worker_prefetch_count if config.backpressure_enabled else 0
    broker = create_broker(config, prefetch_count=prefetch)

    @asynccontextmanager
    async def lifespan(context: ContextRepo) -> AsyncIterator[None]:
        engine = get_engine(config)
        executor = ThreadPoolExecutor(max_workers=config.worker_concurrency)
        try:
            await init_storage(engine)
            model = await asyncio.to_thread(load_model, config)
            context.set_global("engine", engine)
            context.set_global("executor", executor)
            context.set_global("model", model)
            logger.info("worker model loaded", extra={"fields": model.info.model_dump()})
            yield
        finally:
            await asyncio.to_thread(executor.shutdown, wait=True)
            await engine.dispose()

    app = AsgiFastStream(
        broker,
        lifespan=lifespan,
        asgi_routes=[("/healthz", make_ping_asgi(broker, timeout=2.0))],
    )

    @app.on_startup
    async def prepare_queues() -> None:
        await broker.connect()
        await declare_queues(broker, config)

    async def dead_letter(message: RabbitMessage, reason: str) -> None:
        await publish_with_retry(
            broker,
            config,
            message.body,
            queue=f"{config.rabbitmq_queue}.dlq",
            message_id=message.message_id,
            headers={**message.headers, "failure": reason[:1000]},
        )
        logger.warning("task sent to DLQ", extra={"fields": {"reason": reason[:1000]}})

    @broker.subscriber(
        get_task_queue(config),
        ack_policy=AckPolicy.NACK_ON_ERROR,
        decoder=lambda message: message.body,
        no_reply=True,
    )
    async def handle_task(
        message: RabbitMessage,
        engine: Annotated[AsyncEngine, Context()],
        executor: Annotated[ThreadPoolExecutor, Context()],
        model: Annotated[LoadedModel, Context()],
    ) -> None:
        try:
            task = TaskMessage.model_validate_json(message.body)
        except ValidationError as error:
            await dead_letter(message, str(error))
            return

        response = None
        attempt = 0
        while True:
            try:
                if await get_result(engine, task.task_id) is not None:
                    logger.info(
                        "task already completed", extra={"fields": {"task_id": str(task.task_id)}}
                    )
                    return
                if response is None:
                    attempt = await next_attempt(engine, task.task_id)
                    if attempt > config.max_attempts:
                        await dead_letter(message, "attempt limit reached (including redeliveries)")
                        return
                    try:
                        response = await asyncio.get_running_loop().run_in_executor(
                            executor, predict, task, model, config.inference_delay
                        )
                    except Exception as error:
                        logger.exception("inference failed", extra={"fields": {"attempt": attempt}})
                        if attempt >= config.max_attempts:
                            await dead_letter(message, str(error))
                            return
                        await asyncio.sleep(config.retry_delay)
                        continue
                await save_result(engine, task.task_id, response)
                logger.info(
                    "task completed",
                    extra={
                        "fields": {
                            "task_id": str(task.task_id),
                            "attempt": attempt,
                            "processing_ms": response.processing_ms,
                            "total_ms": response.total_ms,
                        }
                    },
                )
                return
            except (SQLAlchemyError, OSError, TimeoutError):
                logger.exception("storage unavailable; retrying")
                await asyncio.sleep(config.retry_delay)

    return app


def worker_factory() -> AsgiFastStream:
    config = get_config()
    setup_logging(config.log_level)
    return create_worker(config)
