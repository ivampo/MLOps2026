import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import version as package_version

from fastapi import FastAPI, Request, Response
from opentelemetry import propagate, trace

from mlops.api import healthz, process, v1
from mlops.broker import create_broker, declare_queues
from mlops.config import get_config
from mlops.db import get_engine
from mlops.model import ModelCache, load_model
from mlops.tasks import create_table
from mlops.telemetry import instruments, setup_telemetry, tracer

logger = logging.getLogger(__name__)


access_logger = logging.getLogger("mlops.access")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = get_config()
    setup_telemetry("api", config.inference_delay)
    app.state.engine = get_engine(config)
    try:
        app.state.model_cache = ModelCache(config, await asyncio.to_thread(load_model, config))
        if config.pipeline_enabled:
            await create_table(app.state.engine)
            app.state.broker = create_broker(config)
            await app.state.broker.connect()
            await declare_queues(app.state.broker, config)
        logger.info(
            "app started",
            extra={
                "fields": {
                    "version": package_version("mlops"),
                    "log_level": config.log_level,
                    "postgres_host": config.postgres_host,
                    "postgres_db": config.postgres_db,
                    "model": app.state.model_cache.active.info.model_dump(),
                }
            },
        )
        yield
    finally:
        logger.info("graceful shutdown started")
        if hasattr(app.state, "broker"):
            await app.state.broker.close()
            del app.state.broker
        if hasattr(app.state, "model_cache"):
            del app.state.model_cache
        await app.state.engine.dispose()
        logger.info("engine disposed")
        logger.info("graceful shutdown completed")


app = FastAPI(lifespan=lifespan)
app.include_router(v1.router)
app.include_router(healthz.router)
app.include_router(process.router)


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    started = time.perf_counter()
    with tracer.start_as_current_span(
        f"{request.method} {request.url.path}",
        context=propagate.extract(request.headers), kind=trace.SpanKind.SERVER,
    ) as span:
        try:
            response = await call_next(request)
        except Exception:
            route = request.scope.get("route")
            path = route.path if route else "unmatched"
            elapsed = time.perf_counter() - started
            instruments.api_requests.add(1, {"route": path, "status": 500})
            instruments.api_duration.record(elapsed, {"route": path})
            access_logger.error("request failed", extra={"fields": {"path": path, "status": 500}})
            raise
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        span.update_name(f"{request.method} {path}")
        span.set_attribute("http.response.status_code", response.status_code)
        if response.status_code >= 500:
            span.set_status(trace.StatusCode.ERROR)
        attributes = {"route": path, "status": response.status_code}
        elapsed = time.perf_counter() - started
        instruments.api_requests.add(1, attributes)
        instruments.api_duration.record(elapsed, {"route": path})
        log = access_logger.debug if path in ("/healthz", "/api/v1/health") else access_logger.info
        if response.status_code >= 500:
            log = access_logger.warning
        log(
            "request",
            extra={
                "fields": {
                    "method": request.method,
                    "path": path,
                    "status": response.status_code,
                    "duration_ms": round(elapsed * 1000, 2),
                }
            },
        )
        return response
