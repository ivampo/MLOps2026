import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from mlops.api import healthz, v1
from mlops.config import get_config
from mlops.db import get_engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = get_config()
    app.state.engine = get_engine(config)
    yield
    await app.state.engine.dispose()
    logger.info("engine disposed")


app = FastAPI(lifespan=lifespan)
app.include_router(v1.router)
app.include_router(healthz.router)
