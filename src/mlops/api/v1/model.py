from fastapi import APIRouter, Request

from mlops.schemas import ModelInfo

router = APIRouter(tags=["inference"])


@router.get("/model")
async def model(request: Request) -> ModelInfo:
    return request.app.state.model.info
