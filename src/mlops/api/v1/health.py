import asyncio
import logging

from fastapi import APIRouter, Request, Response, status

from mlops.config import get_config
from mlops.db import ping
from mlops.schemas import HealthComponent, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request, response: Response) -> HealthResponse:
    config = get_config()
    try:
        async with asyncio.timeout(config.health_timeout):
            pg_version, elapsed_ms = await ping(request.app.state.engine)
    except Exception as e:
        logger.warning("postgres ping failed", exc_info=e)
        postgres = HealthComponent(name="postgres", health=False, error=str(e))
    else:
        postgres = HealthComponent(
            name="postgres", health=True, version=pg_version, response_time=elapsed_ms
        )

    components = [postgres]
    healthy = all(component.health for component in components)
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(health=healthy, components=components)
