import asyncio

import pytest
from httpx import AsyncClient
from sqlalchemy.exc import OperationalError

from mlops.config import get_config


async def test_health_working(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_ping(engine: object) -> tuple[str, float]:
        return "228", 6.66

    monkeypatch.setattr("mlops.api.v1.health.ping", mock_ping)
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "health": True,
        "components": [
            {
                "name": "postgres",
                "health": True,
                "version": "228",
                "response_time": 6.66,
                "error": None,
            }
        ],
    }


async def test_health_db_down(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_ping(engine: object) -> tuple[str, float]:
        raise ConnectionRefusedError("error")

    monkeypatch.setattr("mlops.api.v1.health.ping", mock_ping)
    response = await client.get("/api/v1/health")
    response_body = response.json()
    assert response.status_code == 503
    assert not response_body["health"]
    assert not response_body["components"][0]["health"]
    assert response_body["components"][0]["error"] == "error"
    assert response_body["components"][0]["response_time"] is None
    assert response_body["components"][0]["version"] is None


async def test_health_db_timeout(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_ping(engine: object) -> tuple[str, float]:
        await asyncio.sleep(10)
        return "228", 10000

    monkeypatch.setattr(get_config(), "health_timeout", 0.1)
    monkeypatch.setattr("mlops.api.v1.health.ping", mock_ping)
    response = await client.get("/api/v1/health")
    response_body = response.json()
    assert response.status_code == 503
    assert not response_body["health"]


async def test_health_db_error(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_ping(engine: object) -> tuple[str, float]:
        raise OperationalError("SHOW server_version", None, Exception("auth failed"))

    monkeypatch.setattr("mlops.api.v1.health.ping", mock_ping)
    response = await client.get("/api/v1/health")
    response_body = response.json()
    assert response.status_code == 503
    assert not response_body["health"]
