import logging
from importlib.metadata import version as package_version

import mlflow
import mlflow.sklearn
import pandas as pd
from matplotlib.figure import Figure
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops.config import get_config
from mlops.data import CLASSES, DATA_SOURCE, FEATURES, load_dataset
from mlops.logging import setup_logging

logger = logging.getLogger(__name__)
SEED = 42


def log_dataset(frame: pd.DataFrame, name: str, context: str) -> None:
    dataset = mlflow.data.from_pandas(frame, source=DATA_SOURCE, targets="target", name=name)
    mlflow.log_input(dataset, context=context)


def log_eda(frame: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> str:
    with mlflow.start_run(run_name="eda") as run:
        log_dataset(frame, "iris", "eda")
        mlflow.log_param("sklearn_version", package_version("scikit-learn"))
        mlflow.log_dict(
            {
                "rows": len(frame),
                "missing": frame.isna().sum().to_dict(),
                "duplicates": int(frame.duplicated().sum()),
                "class_counts": frame["target"].value_counts().to_dict(),
            },
            "eda/summary.json",
        )
        mlflow.log_text(frame.describe().to_csv(), "eda/statistics.csv")
        mlflow.log_text(frame.to_csv(index=True, index_label="row_id"), "dataset/iris.csv")
        mlflow.log_dict(
            {name: part.index.tolist() for name, part in splits.items()}, "dataset/split.json"
        )

        figure = Figure(figsize=(6, 4), layout="constrained")
        axes = figure.subplots()
        counts = frame["target"].value_counts().reindex(CLASSES)
        axes.bar(counts.index, counts.values)
        axes.set(title="Class balance", ylabel="Samples")
        mlflow.log_figure(figure, "eda/class_balance.png")

        figure = Figure(figsize=(6, 4), layout="constrained")
        axes = figure.subplots()
        for label, group in frame.groupby("target"):
            axes.scatter(group["petal_length"], group["petal_width"], label=label, alpha=0.7)
        axes.set(xlabel="Petal length, cm", ylabel="Petal width, cm", title="Petal separation")
        axes.legend()
        mlflow.log_figure(figure, "eda/petals.png")

        figure = Figure(figsize=(7, 5), layout="constrained")
        axes = figure.subplots()
        correlations = frame[FEATURES].corr()
        image = axes.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm")
        axes.set(xticks=range(4), yticks=range(4), xticklabels=FEATURES, yticklabels=FEATURES)
        axes.tick_params(axis="x", labelrotation=30)
        figure.colorbar(image, ax=axes)
        axes.set_title("Feature correlations")
        mlflow.log_figure(figure, "eda/correlations.png")

        mlflow.log_text(
            "# Iris: краткий EDA\n\n"
            "150 наблюдений, четыре числовых признака в сантиметрах, три класса по 50 строк.\n"
            "Пропусков нет; один полный дубликат удалён перед разбиением.\n"
            "По длине и ширине лепестка setosa хорошо отделяется, а versicolor и virginica "
            "частично пересекаются. Размеры лепестка сильно коррелируют.\n"
            "Проверяем линейную модель с масштабированием и нелинейный случайный лес. "
            "Scaler обучается только на train внутри Pipeline.\n"
            "Основная метрика — macro F1: каждый вид имеет одинаковый вес. "
            "Accuracy показывает долю правильных ответов, log loss — качество вероятностей.\n"
            "Модель выбирается на validation; test используется только после выбора. "
            "Небольшой учебный датасет не позволяет делать выводы о работе на новых популяциях.\n",
            "eda/notes.md",
        )
        return run.info.run_id


def evaluate(model: Pipeline, frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    features, target = frame[FEATURES], frame["target"]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    metrics = {
        f"{prefix}_f1_macro": f1_score(target, predictions, average="macro", zero_division=0),
        f"{prefix}_accuracy": accuracy_score(target, predictions),
        f"{prefix}_log_loss": log_loss(target, probabilities, labels=model.classes_),
    }
    mlflow.log_metrics(metrics)
    mlflow.log_dict(
        classification_report(target, predictions, output_dict=True, zero_division=0),
        f"{prefix}/classification_report.json",
    )
    figure = Figure(figsize=(6, 5), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        target, predictions, labels=CLASSES, ax=figure.subplots(), colorbar=False
    )
    mlflow.log_figure(figure, f"{prefix}/confusion_matrix.png")
    diagnostics = frame.assign(prediction=predictions)
    mlflow.log_text(
        diagnostics.to_csv(index=True, index_label="row_id"), f"{prefix}/predictions.csv"
    )
    return metrics


def train() -> dict[str, str]:
    config = get_config()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)
    frame = load_dataset()
    clean = frame.drop_duplicates()
    development, test = train_test_split(
        clean, test_size=0.2, stratify=clean["target"], random_state=SEED
    )
    training, validation = train_test_split(
        development, test_size=0.25, stratify=development["target"], random_state=SEED
    )
    eda_run_id = log_eda(frame, {"train": training, "validation": validation, "test": test})
    variants = {
        "baseline": Pipeline([("classifier", DummyClassifier(strategy="prior"))]),
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("classifier", LogisticRegression(C=1.0, max_iter=500))]
        ),
        "random_forest": Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=100, max_depth=3, random_state=SEED),
                )
            ]
        ),
    }
    client = MlflowClient()
    results = []
    for name, model in variants.items():
        with mlflow.start_run(run_name=name) as run:
            log_dataset(frame, "iris", "source")
            log_dataset(training, "iris-train", "training")
            log_dataset(validation, "iris-validation", "validation")
            mlflow.set_tags({"eda_run_id": eda_run_id, "selection_metric": "val_f1_macro"})
            mlflow.log_params(
                {
                    "variant": name,
                    "preprocessing": "standard_scaler" if "scaler" in model.named_steps else "none",
                    "seed": SEED,
                    "train_rows": len(training),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                    "sklearn_version": package_version("scikit-learn"),
                    **model["classifier"].get_params(),
                }
            )
            model.fit(training[FEATURES], training["target"])
            mlflow.log_metric(
                "train_f1_macro",
                f1_score(training["target"], model.predict(training[FEATURES]), average="macro"),
            )
            metrics = evaluate(model, validation, "val")
            example = training[FEATURES].head(3)
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                signature=infer_signature(example, model.predict(example)),
                input_example=example,
                pip_requirements=[
                    f"{package}=={package_version(package)}"
                    for package in ["scikit-learn", "pandas", "numpy", "cloudpickle"]
                ],
            )
            version = None
            if name != "baseline":
                version = str(
                    mlflow.register_model(info.model_uri, config.mlflow_model_name).version
                )
                client.set_model_version_tag(config.mlflow_model_name, version, "variant", name)
            results.append(
                {"variant": name, "run_id": run.info.run_id, "version": version, **metrics}
            )

    candidates = [result for result in results if result["version"] is not None]
    winner = max(candidates, key=lambda result: (result["val_f1_macro"], -result["val_log_loss"]))
    with mlflow.start_run(run_id=winner["run_id"]):
        log_dataset(test, "iris-test", "testing")
        evaluate(variants[winner["variant"]], test, "test")
        mlflow.log_text(pd.DataFrame(results).to_csv(index=False), "comparison.csv")
        mlflow.log_dict(winner, "selection.json")
        mlflow.set_tag("selected", "true")
    client.set_registered_model_alias(
        config.mlflow_model_name, config.mlflow_model_alias, winner["version"]
    )
    logger.info("champion selected", extra={"fields": winner})
    return {
        "name": config.mlflow_model_name,
        "version": winner["version"],
        "run_id": winner["run_id"],
    }


def main() -> None:
    setup_logging(get_config().log_level)
    train()
