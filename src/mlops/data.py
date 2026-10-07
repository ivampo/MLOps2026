from importlib.resources import files

import pandas as pd
from sklearn.datasets import load_iris

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASSES = ["setosa", "versicolor", "virginica"]
DATA_SOURCE = str(files("sklearn.datasets").joinpath("data", "iris.csv"))


def load_dataset() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    frame = iris.frame.copy()
    frame.columns = [*FEATURES, "target"]
    frame["target"] = frame["target"].map(dict(enumerate(CLASSES)))
    return frame
