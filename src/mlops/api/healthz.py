from fastapi import APIRouter

from mlops.schemas import HealthzResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthzResponse)
async def healthz():
    return HealthzResponse(status="ok")
