"""Data source definitions for Iris dataset."""

import pandas as pd
from sklearn.datasets import load_iris

from geronimo.data import DataSource


def _load_iris_dataframe() -> pd.DataFrame:
    """Load Iris dataset from sklearn."""
    iris = load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    df["species"] = iris.target
    df["species_name"] = df["species"].map({
        0: "setosa",
        1: "versicolor", 
        2: "virginica"
    })
    return df


# Training data using the function source pattern
training_data = DataSource(
    name="iris_training",
    source="function",
    handle=_load_iris_dataframe,
)
