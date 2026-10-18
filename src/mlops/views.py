"""Сохранённый вид таблицы Runs для MLflow 3.17."""

import json
import time

from mlflow import MlflowClient

VIEW_KEY = "mlflow.sharedViewState.iris-comparison"


def configure_comparison_view(client: MlflowClient, experiment_id: str) -> None:
    # Preserve existing column layout and other user customizations.
    experiment = client.get_experiment(experiment_id)
    if VIEW_KEY in experiment.tags:
        return
    state = {
        "selectedColumns": [
            "metrics.`val_f1_macro`",
            "metrics.`val_accuracy`",
            "metrics.`val_log_loss`",
            "metrics.`train_f1_macro`",
            "attributes.`Models`",
        ],
        "columnOrder": [],
        "columnWidths": {},
        "searchFilter": "metrics.val_f1_macro >= 0",
        "orderByKey": "metrics.`val_f1_macro`",
        "orderByAsc": False,
        "startTime": "ALL",
        "lifecycleFilter": "Active",
        "modelVersionFilter": "All Runs",
        "datasetsFilter": [],
        "runsHiddenMode": "SHOW_ALL",
    }
    now = int(time.time() * 1000)
    client.set_experiment_tag(
        experiment_id,
        VIEW_KEY,
        json.dumps(
            {
                "name": "Iris: сравнение моделей",
                "createdAt": now,
                "updatedAt": now,
                "state": json.dumps(state),
            },
            ensure_ascii=False,
        ),
    )
