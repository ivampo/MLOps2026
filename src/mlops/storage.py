import asyncio
from uuid import UUID

from sqlalchemy import Column, DateTime, Integer, MetaData, Table, Uuid, func, select, text
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.ext.asyncio import AsyncEngine

from mlops.config import get_config
from mlops.db import get_engine
from mlops.schemas import TaskResult

metadata = MetaData()

task_results = Table(
    "task_results",
    metadata,
    Column("task_id", Uuid(as_uuid=True), primary_key=True),
    Column("result", JSONB, nullable=False),
    Column(
        "completed_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
)

task_attempts = Table(
    "task_attempts",
    metadata,
    Column("task_id", Uuid(as_uuid=True), primary_key=True),
    Column("attempts", Integer, nullable=False),
)


async def init_storage(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.execute(text("SELECT pg_advisory_xact_lock(4774)"))
        await connection.run_sync(metadata.create_all)


async def save_result(
    engine: AsyncEngine,
    task_id: UUID,
    response: TaskResult,
) -> bool:
    statement = (
        insert(task_results)
        .values(
            task_id=task_id,
            result=response.model_dump(mode="json"),
        )
        .on_conflict_do_nothing(index_elements=[task_results.c.task_id])
        .returning(task_results.c.task_id)
    )

    async with engine.begin() as connection:
        inserted_id = await connection.scalar(statement)

    return inserted_id is not None


async def get_result(
    engine: AsyncEngine,
    task_id: UUID,
) -> TaskResult | None:
    statement = select(task_results.c.result).where(task_results.c.task_id == task_id)

    async with engine.connect() as connection:
        payload = await connection.scalar(statement)

    if payload is None:
        return None

    return TaskResult.model_validate(payload)


async def next_attempt(engine: AsyncEngine, task_id: UUID) -> int:
    statement = (
        insert(task_attempts)
        .values(task_id=task_id, attempts=1)
        .on_conflict_do_update(
            index_elements=[task_attempts.c.task_id],
            set_={"attempts": task_attempts.c.attempts + 1},
        )
        .returning(task_attempts.c.attempts)
    )
    async with engine.begin() as connection:
        result = await connection.execute(statement)
        return result.scalar_one()


async def main() -> None:
    engine = get_engine(get_config())
    try:
        await init_storage(engine)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
