import asyncio
import threading
from unittest.mock import AsyncMock, Mock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine

from mlops.main import app
from mlops.model import LoadedModel

SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


async def test_process(client: AsyncClient) -> None:
    response = await client.post("/process", json={"samples": [SAMPLE, SAMPLE]})
    assert response.status_code == 200
    body = response.json()
    assert body["model"]["version"] == "1"
    assert body["model"]["run_id"] == "test-run"
    assert len(body["predictions"]) == 2
    for prediction in body["predictions"]:
        assert prediction["class_name"] == "setosa"
        assert set(prediction["probabilities"]) == {"setosa", "versicolor", "virginica"}
        assert sum(prediction["probabilities"].values()) == pytest.approx(1)


@pytest.mark.parametrize(
    "payload",
    [
        {"samples": []},
        {"samples": [SAMPLE] * 1001},
        {"samples": [{**SAMPLE, "sepal_length": 0}]},
        {"samples": [{**SAMPLE, "petal_width": -1}]},
        {"samples": [{**SAMPLE, "petal_width": "NaN"}]},
        {"samples": [{**SAMPLE, "petal_width": True}]},
        {"samples": [{**SAMPLE, "unknown": 1}]},
        {"samples": [{"sepal_length": 5.1}]},
        {"samples": [SAMPLE], "unknown": 1},
    ],
)
async def test_process_invalid_payload(client: AsyncClient, payload: dict) -> None:
    assert (await client.post("/process", json=payload)).status_code == 422


async def test_model_loaded_once(
    monkeypatch: pytest.MonkeyPatch, loaded_model: LoadedModel
) -> None:
    loader = Mock(return_value=loaded_model)
    monkeypatch.setattr("mlops.main.load_model", loader)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        for _ in range(3):
            response = await client.post("/process", json={"samples": [SAMPLE]})
            assert response.status_code == 200
        assert (await client.get("/api/v1/model")).json() == loaded_model.info.model_dump()
    loader.assert_called_once()


async def test_inference_runs_outside_event_loop(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, loaded_model: LoadedModel
) -> None:
    event_loop_thread = threading.get_ident()
    original_predict = loaded_model.predict
    worker_threads = []

    def predict(samples):
        worker_threads.append(threading.get_ident())
        return original_predict(samples)

    monkeypatch.setattr(loaded_model, "predict", predict)
    response = await client.post("/process", json={"samples": [SAMPLE]})
    assert response.status_code == 200
    assert worker_threads
    assert worker_threads[0] != event_loop_thread


async def test_startup_failure_disposes_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = AsyncMock(spec=AsyncEngine)
    monkeypatch.setattr("mlops.main.get_engine", lambda config: engine)
    monkeypatch.setattr("mlops.main.load_model", Mock(side_effect=RuntimeError("alias missing")))
    with pytest.raises(RuntimeError, match="alias missing"):
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0)
    engine.dispose.assert_awaited_once()
