import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import version as package_version

from fastapi import FastAPI, Request, Response

from mlops.api import healthz, v1
from mlops.config import get_config
from mlops.db import get_engine

logger = logging.getLogger(__name__)


access_logger = logging.getLogger("mlops.access")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = get_config()
    app.state.engine = get_engine(config)
    logger.info(
        "app started",
        extra={
            "fields": {
                "version": package_version("mlops"),
                "log_level": config.log_level,
                "postgres_host": config.postgres_host,
                "postgres_db": config.postgres_db,
            }
        },
    )
    yield
    await app.state.engine.dispose()
    logger.info("engine disposed")


app = FastAPI(lifespan=lifespan)
app.include_router(v1.router)
app.include_router(healthz.router)


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    started = time.perf_counter()
    response = await call_next(request)
    access_logger.info(
        "request",
        extra={
            "fields": {
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        },
    )
    return response
