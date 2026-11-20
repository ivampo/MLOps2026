import asyncio

from fastapi import APIRouter, Request

from mlops.schemas import ProcessRequest, ProcessResponse
from mlops.telemetry import measured_inference

router = APIRouter(tags=["inference"])


@router.post("/process")
async def process(payload: ProcessRequest, request: Request) -> ProcessResponse:
    model = request.app.state.model_cache.active
    def predict():
        with measured_inference(model):
            return model.predict(payload.samples)

    predictions = await asyncio.to_thread(predict)
    return ProcessResponse(model=model.info, predictions=predictions)
