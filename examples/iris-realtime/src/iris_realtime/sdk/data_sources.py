"""Data source definitions for Iris dataset."""

from geronimo.data import DataSource

# Training data - uses sklearn's built-in iris dataset
# In production, this would point to Snowflake/Postgres/S3
training_data = DataSource(
    name="iris_training",
    source="file",
    path="data/iris_train.csv",
)

# For demonstration, we'll load directly from sklearn
def load_iris_data():
    """Load Iris dataset from sklearn for training/testing."""
    from sklearn.datasets import load_iris
    import pandas as pd
    
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
