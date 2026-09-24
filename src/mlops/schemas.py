from pydantic import BaseModel


class HealthzResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    version: str


class HealthComponent(BaseModel):
    name: str
    health: bool
    version: str | None = None
    response_time: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    health: bool
    components: list[HealthComponent]
