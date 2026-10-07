from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


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


Measurement = Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]


class IrisFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sepal_length: Measurement
    sepal_width: Measurement
    petal_length: Measurement
    petal_width: Measurement


class ProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: list[IrisFeatures] = Field(min_length=1, max_length=1000)


class ModelInfo(BaseModel):
    name: str
    version: str
    alias: str
    run_id: str


class Prediction(BaseModel):
    class_name: str
    probabilities: dict[str, float]


class ProcessResponse(BaseModel):
    model: ModelInfo
    predictions: list[Prediction]
