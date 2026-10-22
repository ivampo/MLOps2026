import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from aio_pika.exceptions import DeliveryError
from httpx import AsyncClient
from pamqp.commands import Basic

from mlops.config import get_config
from mlops.main import app
from mlops.schemas import TaskMessage, TaskResult

SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


async def test_submit_and_get_result(client, monkeypatch, loaded_model):
    task_id = uuid4()
    response = await client.post(
        "/api/v1/tasks", json={"samples": [SAMPLE]}, headers={"Idempotency-Key": str(task_id)}
    )
    assert response.status_code == 202
    assert response.json() == {"task_id": str(task_id), "status": "QUEUED"}
    assert response.headers["Location"] == f"/api/v1/tasks/{task_id}"
    published = app.state.broker.publish.call_args
    task = TaskMessage.model_validate(published.args[0])
    assert task.task_id == task_id
    assert task.submitted_at <= datetime.now(UTC)
    assert published.kwargs["persist"] is True
    result = TaskResult(
        model=loaded_model.info,
        predictions=loaded_model.predict(task.samples),
        processing_ms=1,
        total_ms=2,
    )
    monkeypatch.setattr("mlops.api.v1.tasks.get_result", AsyncMock(return_value=result))
    response = await client.get(f"/api/v1/tasks/{task_id}")
    assert response.status_code == 200
    assert response.json()["result"]["predictions"][0]["class_name"] == "setosa"


async def test_generated_task_id_and_unknown_result(client, monkeypatch):
    response = await client.post("/api/v1/tasks", json={"samples": [SAMPLE]})
    assert response.status_code == 202
    monkeypatch.setattr("mlops.api.v1.tasks.get_result", AsyncMock(return_value=None))
    assert (await client.get(response.headers["Location"])).status_code == 404
    assert (await client.post("/process", json={"samples": [SAMPLE]})).status_code == 200


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
async def test_invalid_task(client: AsyncClient, payload: dict):
    assert (await client.post("/api/v1/tasks", json=payload)).status_code == 422
    app.state.broker.publish.assert_not_awaited()


@pytest.mark.parametrize(
    "error", [DeliveryError(None, Basic.Nack()), ConnectionError(), TimeoutError()]
)
async def test_overload_no_database_write(client, monkeypatch, error):
    app.state.broker.publish.side_effect = error
    save = AsyncMock()
    monkeypatch.setattr("mlops.storage.save_result", save)
    response = await client.post("/api/v1/tasks", json={"samples": [SAMPLE]})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert app.state.broker.publish.await_count == 1
    save.assert_not_awaited()


async def test_api_timeout_is_bounded(client, monkeypatch):
    monkeypatch.setattr(get_config(), "publish_timeout", 0.02)

    async def blocked(*args, **kwargs):
        await asyncio.sleep(10)

    app.state.broker.publish.side_effect = blocked
    async with asyncio.timeout(0.5):
        response = await client.post("/api/v1/tasks", json={"samples": [SAMPLE]})
    assert response.status_code == 503
    assert app.state.publish_slots.empty()


async def test_api_inflight_limit(client):
    app.state.publish_slots = asyncio.Queue(maxsize=1)
    app.state.publish_slots.put_nowait(None)
    response = await client.post("/api/v1/tasks", json={"samples": [SAMPLE]})
    assert response.status_code == 503
    app.state.broker.publish.assert_not_awaited()


async def test_startup_failure_cleanup(monkeypatch):
    engine = AsyncMock()
    broker = AsyncMock()
    monkeypatch.setattr("mlops.main.get_engine", lambda config: engine)
    monkeypatch.setattr("mlops.main.create_broker", lambda config: broker)
    monkeypatch.setattr("mlops.main.init_storage", AsyncMock(side_effect=RuntimeError("db down")))
    with pytest.raises(RuntimeError, match="db down"):
        async with app.router.lifespan_context(app):
            pass
    engine.dispose.assert_awaited_once()
    broker.stop.assert_awaited_once()
