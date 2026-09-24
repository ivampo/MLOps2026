from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from mlops.config import get_config
from mlops.main import app


@pytest.fixture(autouse=True)
def _config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "TEST")
    get_config.cache_clear()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
