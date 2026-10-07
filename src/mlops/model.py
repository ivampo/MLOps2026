from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from mlops.config import Config
from mlops.data import FEATURES
from mlops.schemas import IrisFeatures, ModelInfo, Prediction


@dataclass
class LoadedModel:
    estimator: Pipeline
    info: ModelInfo

    def predict(self, samples: list[IrisFeatures]) -> list[Prediction]:
        frame = pd.DataFrame([sample.model_dump() for sample in samples], columns=FEATURES)
        labels = self.estimator.predict(frame)
        probabilities = self.estimator.predict_proba(frame)
        return [
            Prediction(
                class_name=str(label),
                probabilities={
                    str(name): float(value)
                    for name, value in zip(self.estimator.classes_, scores, strict=True)
                },
            )
            for label, scores in zip(labels, probabilities, strict=True)
        ]


def load_model(config: Config) -> LoadedModel:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient()
    version = client.get_model_version_by_alias(config.mlflow_model_name, config.mlflow_model_alias)
    # Фиксируем версию: alias может измениться во время скачивания модели.
    uri = f"models:/{config.mlflow_model_name}/{version.version}"
    estimator = mlflow.sklearn.load_model(uri)
    info = ModelInfo(
        name=config.mlflow_model_name,
        version=str(version.version),
        alias=config.mlflow_model_alias,
        run_id=version.run_id,
    )
    return LoadedModel(estimator=estimator, info=info)
