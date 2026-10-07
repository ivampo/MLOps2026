from fastapi import APIRouter

from mlops.api.v1 import health, model, version

router = APIRouter(prefix="/api/v1")
router.include_router(health.router)
router.include_router(version.router)
router.include_router(model.router)
