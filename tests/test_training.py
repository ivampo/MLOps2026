import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from mlops.config import get_config
from mlops.train import train


def test_training_registry_and_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Настоящий MLflow и настоящие модели, изолированные от рабочего сервера.
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.chdir(tmp_path)
    selected = train()
    config = get_config()
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(config.mlflow_experiment_name)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 4
    assert all(run.info.status == "FINISHED" for run in runs)
    eda = next(run for run in runs if run.data.tags["mlflow.runName"] == "eda")
    candidates = [run for run in runs if "variant" in run.data.params]
    assert len(candidates) == 3
    for run in candidates:
        assert {"val_f1_macro", "val_accuracy", "val_log_loss"} <= run.data.metrics.keys()
        assert run.data.tags["eda_run_id"] == eda.info.run_id
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
    winner = client.get_run(champion.run_id)
    assert {"test_f1_macro", "test_accuracy", "test_log_loss"} <= winner.data.metrics.keys()
    assert all(
        "test_f1_macro" not in run.data.metrics
        for run in runs
        if run.info.run_id != champion.run_id
    )
    best_f1 = max(
        client.get_run(version.run_id).data.metrics["val_f1_macro"] for version in versions
    )
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

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
