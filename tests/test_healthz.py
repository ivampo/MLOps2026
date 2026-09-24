import pytest
from httpx import AsyncClient


async def test_healthz_return_ok(client: AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_healthz_not_versioned(client: AsyncClient) -> None:
    response = await client.get("/api/v1/healthz")
    assert response.status_code == 404


async def test_healthz_200_dead_database(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def raise_exception(engine: object) -> tuple[str, float]:
        raise ConnectionRefusedError("error")

    monkeypatch.setattr("mlops.api.v1.health.ping", raise_exception)

    assert (await client.get("/healthz")).status_code == 200
