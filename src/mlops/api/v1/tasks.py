import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request

from mlops.broker import publish_task
from mlops.config import get_config
from mlops.schemas import ProcessRequest
from mlops.tasks import get_result
from mlops.telemetry import task_context

router = APIRouter(tags=["tasks"])
logger = logging.getLogger(__name__)


@router.post("/tasks", status_code=202)
async def submit(payload: ProcessRequest, request: Request) -> dict:
    identifier = str(uuid4())
    with task_context(identifier):
        try:
            await publish_task(
                request.app.state.broker, get_config(), payload.model_dump(), identifier
            )
        except Exception as exc:
            logger.warning("task rejected; returning 503")
            raise HTTPException(
                status_code=503, detail="Queue unavailable or full", headers={"Retry-After": "1"}
            ) from exc
    return {"task_id": identifier}


@router.get("/tasks/{identifier}")
async def result(identifier: UUID, request: Request) -> dict:
    value = await get_result(request.app.state.engine, str(identifier))
    if value is None:
        raise HTTPException(status_code=404, detail="Result not available")
    return {"task_id": str(identifier), "result": value}
