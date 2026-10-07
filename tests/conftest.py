from collections.abc import AsyncIterator, Iterator

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
async def client(
    monkeypatch: pytest.MonkeyPatch, loaded_model: LoadedModel
) -> AsyncIterator[AsyncClient]:
    monkeypatch.setattr("mlops.main.load_model", lambda config: loaded_model)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
