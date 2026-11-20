import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from mlops.telemetry import tracer


async def create_table(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.execute(text("""
            CREATE TABLE IF NOT EXISTS task_results (
                task_id TEXT PRIMARY KEY,
                result JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))


async def save_result(engine: AsyncEngine, identifier: str, result: dict) -> None:
    with tracer.start_as_current_span("db.write_result", attributes={"db.system": "postgresql"}):
        async with engine.begin() as connection:
            await connection.execute(
                text("""
                    INSERT INTO task_results (task_id, result)
                    VALUES (:task_id, CAST(:result AS JSONB))
                    ON CONFLICT (task_id) DO NOTHING
                """),
                {"task_id": identifier, "result": json.dumps(result)},
            )


async def get_result(engine: AsyncEngine, identifier: str) -> dict | None:
    with tracer.start_as_current_span("db.read_result"):
        async with engine.connect() as connection:
            return await connection.scalar(
                text("SELECT result FROM task_results WHERE task_id = :task_id"),
                {"task_id": identifier},
            )
