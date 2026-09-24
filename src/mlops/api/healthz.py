from fastapi import APIRouter

from mlops.schemas import HealthzResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthzResponse)
async def liveness():
    return HealthzResponse(status="ok")
