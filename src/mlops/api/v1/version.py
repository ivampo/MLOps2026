from importlib.metadata import version as package_version

from fastapi import APIRouter

from mlops.schemas import VersionResponse

router = APIRouter(tags=["version"])

APP_VERSION = package_version("mlops")


@router.get("/version", response_model=VersionResponse)
async def version():
    return VersionResponse(version=APP_VERSION)
