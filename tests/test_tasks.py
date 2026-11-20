from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

from mlops.main import app

SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


async def test_submit_task(client, monkeypatch):
    publisher = AsyncMock()
    app.state.broker = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr("mlops.api.v1.tasks.publish_task", publisher)
    response = await client.post("/api/v1/tasks", json={"samples": [SAMPLE]})
    assert response.status_code == 202
    identifier = response.json()["task_id"]
    assert str(UUID(identifier)) == identifier
    assert publisher.call_args.args[2] == {"samples": [SAMPLE]}
    assert publisher.call_args.args[3] == identifier


async def test_full_queue_returns_503_with_retry_after(client, monkeypatch):
    app.state.broker = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(
        "mlops.api.v1.tasks.publish_task", AsyncMock(side_effect=TimeoutError("unavailable"))
    )
    response = await client.post("/api/v1/tasks", json={"samples": [SAMPLE]})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"


async def test_read_finished_or_missing_task(client, monkeypatch):
    identifier = "c174a80c-6a03-4ad8-8f3d-ac8b9f5ee4d3"
    reader = AsyncMock(side_effect=[{"predictions": []}, None])
    monkeypatch.setattr("mlops.api.v1.tasks.get_result", reader)
    response = await client.get(f"/api/v1/tasks/{identifier}")
    assert response.status_code == 200
    assert response.json()["result"] == {"predictions": []}
    assert (await client.get(f"/api/v1/tasks/{identifier}")).status_code == 404
