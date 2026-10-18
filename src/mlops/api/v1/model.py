import logging

from fastapi import APIRouter, HTTPException, Request
from mlflow.exceptions import MlflowException

from mlops.schemas import ModelCacheResponse, ModelInfo, ModelSwitchRequest, ModelSwitchResponse

router = APIRouter(tags=["inference"])
logger = logging.getLogger(__name__)


@router.get("/model")
async def model(request: Request) -> ModelInfo:
    return request.app.state.model_cache.active.info


@router.get("/model/cache")
async def model_cache(request: Request) -> ModelCacheResponse:
    cache = request.app.state.model_cache
    return ModelCacheResponse(
        active=cache.active.info, versions=cache.versions, capacity=cache.config.model_cache_size
    )


@router.post("/model/switch")
async def switch_model(payload: ModelSwitchRequest, request: Request) -> ModelSwitchResponse:
    try:
        loaded, cache_hit = await request.app.state.model_cache.switch(payload.version)
    except MlflowException as exc:
        logger.warning("model switch failed", extra={"fields": {"version": payload.version}})
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            raise HTTPException(status_code=404, detail="Model version not found") from exc
        raise HTTPException(status_code=503, detail="Cannot load model from MLflow") from exc
    logger.info(
        "model switched",
        extra={"fields": {"version": loaded.info.version, "cache_hit": cache_hit}},
    )
    return ModelSwitchResponse(model=loaded.info, cache_hit=cache_hit)
