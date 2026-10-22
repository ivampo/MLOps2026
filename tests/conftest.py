from collections.abc import AsyncIterator, Iterator
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops.config import get_config
from mlops.data import FEATURES, load_dataset
from mlops.main import app
from mlops.model import LoadedModel
from mlops.schemas import ModelInfo


@pytest.fixture(autouse=True)
def _config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("POSTGRES_PASSWORD", "TEST")
    monkeypatch.setenv("RABBITMQ_PASSWORD", "TEST")
    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture(scope="session")
def loaded_model() -> LoadedModel:
    frame = load_dataset()
    estimator = Pipeline(
        [("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=500))]
    ).fit(frame[FEATURES], frame["target"])
    return LoadedModel(
        estimator=estimator,
        info=ModelInfo(name="iris-classifier", version="1", alias="champion", run_id="test-run"),
    )


@pytest.fixture
def api_dependencies(monkeypatch: pytest.MonkeyPatch, loaded_model: LoadedModel) -> None:
    monkeypatch.setattr("mlops.main.create_broker", lambda config: AsyncMock())
    monkeypatch.setattr("mlops.main.init_storage", AsyncMock())
    monkeypatch.setattr("mlops.main.declare_queues", AsyncMock())
    monkeypatch.setattr("mlops.main.load_model", lambda config: loaded_model)


@pytest.fixture
async def client(api_dependencies) -> AsyncIterator[AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
