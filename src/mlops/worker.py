import asyncio
import logging

from faststream import FastStream
from faststream.rabbit.annotations import RabbitMessage

from mlops.broker import create_broker, declare_queues, queues
from mlops.config import get_config
from mlops.db import get_engine
from mlops.logging import setup_logging
from mlops.model import load_model
from mlops.processing import handle_task
from mlops.tasks import create_table
from mlops.telemetry import setup_telemetry, shutdown_telemetry

logger = logging.getLogger(__name__)


async def run() -> None:
    config = get_config()
    setup_logging(config.log_level)
    setup_telemetry("worker", config.inference_delay)
    engine = get_engine(config)
    broker = create_broker(config)
    app = FastStream(broker)

    @app.on_startup
    async def startup() -> None:
        await broker.connect()
        await declare_queues(broker, config)
        await create_table(engine)
        app_model = await asyncio.to_thread(load_model, config)
        logger.info("model loaded", extra={"fields": app_model.info.model_dump()})

        @broker.subscriber(queues(config)[0])
        async def consume(body, message: RabbitMessage) -> None:
            await handle_task(message, app_model, engine, broker, config)

    @app.after_startup
    async def ready() -> None:
        await declare_queues(broker, config)
        logger.info("worker ready")

    @app.on_shutdown
    async def stopping() -> None:
        logger.info("graceful shutdown started")

    try:
        await app.run()
    finally:
        await engine.dispose()
        logger.info("graceful shutdown completed")
        await asyncio.to_thread(shutdown_telemetry)


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("worker failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
