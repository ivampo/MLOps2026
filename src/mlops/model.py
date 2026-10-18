import asyncio
from collections import OrderedDict
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


def load_model(config: Config, version_number: str | None = None) -> LoadedModel:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient()
    version = (
        client.get_model_version(config.mlflow_model_name, version_number)
        if version_number is not None
        else client.get_model_version_by_alias(config.mlflow_model_name, config.mlflow_model_alias)
    )
    # Фиксируем версию: alias может измениться во время скачивания модели.
    uri = f"models:/{config.mlflow_model_name}/{version.version}"
    estimator = mlflow.sklearn.load_model(uri)
    info = ModelInfo(
        name=config.mlflow_model_name,
        version=str(version.version),
        alias=config.mlflow_model_alias if version_number is None else None,
        run_id=version.run_id,
    )
    return LoadedModel(estimator=estimator, info=info)


class ModelCache:
    """Process-local LRU cache with serialized loads and an atomic active reference."""

    def __init__(self, config: Config, initial: LoadedModel) -> None:
        self.config = config
        self.active = initial
        self._models = OrderedDict(
            {
                initial.info.version: LoadedModel(
                    estimator=initial.estimator,
                    info=initial.info.model_copy(update={"alias": None}),
                )
            }
        )
        self._lock = asyncio.Lock()

    @property
    def versions(self) -> list[str]:
        return list(reversed(self._models))

    async def switch(self, version: str) -> tuple[LoadedModel, bool]:
        async with self._lock:
            model = self._models.get(version)
            cache_hit = model is not None
            if model is None:
                # Complete loading before changing either the cache or the active model.
                model = await asyncio.to_thread(load_model, self.config, version)
                self._models[version] = model
            self._models.move_to_end(version)
            self.active = model
            while len(self._models) > self.config.model_cache_size:
                self._models.popitem(last=False)
            return model, cache_hit
