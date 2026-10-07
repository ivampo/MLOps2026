import asyncio

from fastapi import APIRouter, Request

from mlops.schemas import ProcessRequest, ProcessResponse

router = APIRouter(tags=["inference"])


@router.post("/process")
async def process(payload: ProcessRequest, request: Request) -> ProcessResponse:
    model = request.app.state.model
    predictions = await asyncio.to_thread(model.predict, payload.samples)
    return ProcessResponse(model=model.info, predictions=predictions)
