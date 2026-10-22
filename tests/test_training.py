import asyncio
import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from mlops.config import get_config
from mlops.data import FEATURES
from mlops.model import ModelCache, load_model
from mlops.train import train
from mlops.views import VIEW_KEY, configure_comparison_view


def test_training_registry_and_alias_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.chdir(tmp_path)
    selected = train()
    config = get_config()
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(config.mlflow_experiment_name)
    assert experiment is not None
    assert VIEW_KEY in experiment.tags
    initial_view = experiment.tags[VIEW_KEY]
    configure_comparison_view(client, experiment.experiment_id)
    stored_experiment = client.get_experiment(experiment.experiment_id)
    assert stored_experiment is not None
    assert stored_experiment.tags[VIEW_KEY] == initial_view
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 4
    assert all(run.info.status == "FINISHED" for run in runs)
    eda = next(run for run in runs if run.data.tags["mlflow.runName"] == "eda")
    candidates = [run for run in runs if "variant" in run.data.params]
    assert len(candidates) == 3
    for run in candidates:
        assert {"val_f1_macro", "val_accuracy", "val_log_loss"} <= run.data.metrics.keys()
        assert run.data.tags["eda_run_id"] == eda.info.run_id
        assert run.inputs is not None
        inputs = run.inputs.dataset_inputs
        assert {item.dataset.name for item in inputs} >= {"iris", "iris-train", "iris-validation"}
        assert all(item.dataset.digest and item.dataset.source for item in inputs)

    versions = client.search_model_versions(f"name='{config.mlflow_model_name}'")
    assert len(versions) == 2
    champion = client.get_model_version_by_alias(
        config.mlflow_model_name, config.mlflow_model_alias
    )
    assert str(champion.version) == selected["version"]
    assert champion.run_id == selected["run_id"]
    assert champion.run_id is not None
    winner = client.get_run(champion.run_id)
    assert {"test_f1_macro", "test_accuracy", "test_log_loss"} <= winner.data.metrics.keys()
    assert all(
        "test_f1_macro" not in run.data.metrics
        for run in runs
        if run.info.run_id != champion.run_id
    )
    scores = []
    for version in versions:
        assert version.run_id is not None
        scores.append(client.get_run(version.run_id).data.metrics["val_f1_macro"])
    best_f1 = max(scores)
    assert winner.data.metrics["val_f1_macro"] == best_f1

    artifacts = {item.path for item in client.list_artifacts(eda.info.run_id, "eda")}
    assert {
        "eda/notes.md",
        "eda/petals.png",
        "eda/class_balance.png",
        "eda/correlations.png",
    } <= artifacts
    split_path = client.download_artifacts(eda.info.run_id, "dataset/split.json", str(tmp_path))
    split = json.loads(Path(split_path).read_text())
    train_ids, val_ids, test_ids = (set(split[name]) for name in ["train", "validation", "test"])
    assert len(train_ids) == 89
    assert len(val_ids) == len(test_ids) == 30
    assert not train_ids & val_ids
    assert not train_ids & test_ids
    assert not val_ids & test_ids
    csv_path = client.download_artifacts(eda.info.run_id, "dataset/iris.csv", str(tmp_path))
    frame = pd.read_csv(csv_path, index_col="row_id")
    assert not frame.loc[list(train_ids | val_ids | test_ids)].duplicated().any()

    other_version = next(
        version.version for version in versions if version.version != champion.version
    )
    sample = frame[FEATURES].iloc[0].to_dict()
    loaded = load_model(config)
    from mlops.schemas import IrisFeatures

    assert loaded.info.version == str(champion.version)
    assert len(loaded.predict([IrisFeatures.model_validate(sample)])) == 1
    cache = ModelCache(config, loaded)

    async def check_switch():
        switched, hit = await cache.switch(str(other_version))
        assert hit is False
        assert switched.info.alias is None
        assert switched.info.version == str(other_version)
        restored, hit = await cache.switch(str(champion.version))
        assert hit is True
        assert restored.info.version == str(champion.version)

    asyncio.run(check_switch())
    assert (
        client.get_model_version_by_alias(
            config.mlflow_model_name, config.mlflow_model_alias
        ).version
        == champion.version
    )
    assert len(client.search_model_versions(f"name='{config.mlflow_model_name}'")) == 2
    client.set_registered_model_alias(
        config.mlflow_model_name, config.mlflow_model_alias, other_version
    )
    assert loaded.info.version == str(champion.version)
    assert load_model(config).info.version == str(other_version)
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
