import asyncio
import math
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Header, HTTPException, Request, Response

from mlops.config import get_config
from mlops.messaging import PUBLISH_ERRORS
from mlops.schemas import ProcessRequest, TaskAccepted, TaskCompleted, TaskMessage
from mlops.storage import get_result

router = APIRouter(tags=["tasks"])


@router.post("/tasks", status_code=202)
async def submit_task(
    payload: ProcessRequest,
    request: Request,
    response: Response,
    idempotency_key: Annotated[UUID | None, Header()] = None,
) -> TaskAccepted:
    config = get_config()
    unavailable = HTTPException(
        status_code=503,
        detail="Queue unavailable or full; retry with the same Idempotency-Key",
        headers={"Retry-After": str(max(1, math.ceil(config.inference_delay)))},
    )
    try:
        request.app.state.publish_slots.put_nowait(None)
    except asyncio.QueueFull:
        raise unavailable from None
    try:
        task = TaskMessage(
            task_id=idempotency_key or uuid4(),
            submitted_at=datetime.now(UTC),
            samples=payload.samples,
        )
        try:
            async with asyncio.timeout(config.publish_timeout):
                await request.app.state.broker.publish(
                    task.model_dump(mode="json"),
                    queue=config.rabbitmq_queue,
                    persist=True,
                    mandatory=True,
                    message_id=str(task.task_id),
                    timeout=config.publish_timeout,
                )
        except PUBLISH_ERRORS:
            raise unavailable from None
        response.headers["Location"] = f"/api/v1/tasks/{task.task_id}"
        return TaskAccepted(task_id=task.task_id)
    finally:
        request.app.state.publish_slots.get_nowait()


@router.get("/tasks/{task_id}")
async def read_result(task_id: UUID, request: Request) -> TaskCompleted:
    result = await get_result(request.app.state.engine, task_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail="Result not ready or task unknown; check DLQ on failure"
        )
    return TaskCompleted(task_id=task_id, result=result)
