import asyncio
import gc
import threading
import weakref
from copy import deepcopy
from unittest.mock import Mock

import pytest
from httpx import AsyncClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

from mlops.config import get_config
from mlops.main import app
from mlops.model import LoadedModel, ModelCache

SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


def version_of(model: LoadedModel, version: str) -> LoadedModel:
    return LoadedModel(
        estimator=model.estimator,
        info=model.info.model_copy(
            update={"version": version, "run_id": f"run-{version}", "alias": None}
        ),
    )


async def test_concurrent_switch_loads_once(monkeypatch, loaded_model):
    cache = ModelCache(get_config(), loaded_model)
    loader = Mock(return_value=version_of(loaded_model, "2"))
    monkeypatch.setattr("mlops.model.load_model", loader)
    first, second = await asyncio.gather(cache.switch("2"), cache.switch("2"))
    assert first[1] is False
    assert second[1] is True
    loader.assert_called_once()
    assert cache.active.info.version == "2"
    assert cache.versions == ["2", "1"]


async def test_lru_evicts_least_recently_selected(monkeypatch, loaded_model):
    config = get_config().model_copy(update={"model_cache_size": 2})
    cache = ModelCache(config, loaded_model)
    loader = Mock(side_effect=lambda config, version: version_of(loaded_model, version))
    monkeypatch.setattr("mlops.model.load_model", loader)
    await cache.switch("2")
    assert (await cache.switch("1"))[1] is True
    await cache.switch("3")
    assert cache.versions == ["3", "1"]
    assert cache.active.info.version == "3"
    assert (await cache.switch("2"))[1] is False
    assert cache.versions == ["2", "3"]
    assert [call.args[1] for call in loader.call_args_list] == ["2", "3", "2"]


@pytest.mark.parametrize(
    ("error_code", "status"),
    [(RESOURCE_DOES_NOT_EXIST, 404), (INTERNAL_ERROR, 503)],
)
async def test_failed_switch_keeps_active_model(client, monkeypatch, error_code, status):
    before = (await client.get("/api/v1/model/cache")).json()
    monkeypatch.setattr(
        "mlops.model.load_model",
        Mock(side_effect=MlflowException("unavailable", error_code=error_code)),
    )
    response = await client.post("/api/v1/model/switch", json={"version": "999"})
    assert response.status_code == status
    assert (await client.get("/api/v1/model/cache")).json() == before
    assert (await client.post("/process", json={"samples": [SAMPLE]})).status_code == 200


@pytest.mark.parametrize("version", ["0", "-1", "champion", "1/../../", 2])
async def test_switch_rejects_invalid_version(client, version):
    response = await client.post("/api/v1/model/switch", json={"version": version})
    assert response.status_code == 422


async def test_inflight_request_keeps_its_model(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, loaded_model: LoadedModel
):
    started = asyncio.Event()
    finish = threading.Event()
    loop = asyncio.get_running_loop()
    original = loaded_model.predict

    def slow_predict(samples):
        loop.call_soon_threadsafe(started.set)
        if not finish.wait(timeout=5):
            raise TimeoutError("Test did not release prediction")
        return original(samples)

    monkeypatch.setattr(loaded_model, "predict", slow_predict)
    monkeypatch.setattr("mlops.model.load_model", Mock(return_value=version_of(loaded_model, "2")))
    pending = asyncio.create_task(client.post("/process", json={"samples": [SAMPLE]}))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        response = await client.post("/api/v1/model/switch", json={"version": "2"})
        assert response.status_code == 200
        assert response.json()["cache_hit"] is False
        assert response.json()["model"]["alias"] is None
        assert (await client.get("/api/v1/model")).json()["version"] == "2"
    finally:
        finish.set()
        old_response = await pending
    assert old_response.json()["model"]["version"] == "1"
    assert app.state.model_cache.active.info.version == "2"
    cached = await client.post("/api/v1/model/switch", json={"version": "1"})
    assert cached.json()["cache_hit"] is True


async def test_evicted_startup_model_is_released(monkeypatch, loaded_model):
    monkeypatch.setenv("MODEL_CACHE_SIZE", "1")
    references = []

    def initial_loader(config):
        initial = deepcopy(loaded_model)
        references.append(weakref.ref(initial.estimator))
        return initial

    monkeypatch.setattr("mlops.main.load_model", initial_loader)
    monkeypatch.setattr(
        "mlops.model.load_model", lambda config, version: version_of(loaded_model, version)
    )
    async with app.router.lifespan_context(app):
        assert references[0]() is not None
        await app.state.model_cache.switch("2")
        gc.collect()
        assert references[0]() is None
    assert not hasattr(app.state, "model_cache")
