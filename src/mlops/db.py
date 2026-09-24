from time import perf_counter

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from mlops.config import Config


def get_engine(config: Config) -> AsyncEngine:
    return create_async_engine(
        config.postgres_dsn, pool_pre_ping=True, connect_args={"timeout": config.db_conn_timeout}
    )


async def ping(engine: AsyncEngine) -> tuple[str, float]:
    started = perf_counter()
    async with engine.connect() as conn:
        version = await conn.scalar(text("SHOW server_version"))
    elapsed_ms = (perf_counter() - started) * 1000
    return str(version), round(elapsed_ms, 2)
