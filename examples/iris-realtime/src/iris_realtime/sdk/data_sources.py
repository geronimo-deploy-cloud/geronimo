"""Data source definitions for Iris dataset.

Uses the new `source="function"` pattern which allows wrapping existing
data loading functions as declarative DataSources.
"""

import pandas as pd
from sklearn.datasets import load_iris

from geronimo.data import DataSource


def _load_iris_dataframe() -> pd.DataFrame:
    """Load Iris dataset from sklearn as a DataFrame.
    
    This function demonstrates wrapping an existing data loading
    function for use with Geronimo's DataSource abstraction.
    
    Returns:
        DataFrame with iris measurements and species labels.
        
    Note:
        When using `source="function"`, the handle MUST return a 
        pandas DataFrame. This is validated at runtime.
    """
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


# =============================================================================
# DataSource Definitions
# =============================================================================

# Training data using the function source pattern
# The handle function is validated at runtime to ensure it returns a DataFrame
training_data = DataSource(
    name="iris_training",
    source="function",
    handle=_load_iris_dataframe,
)

# Alternative: File-based source for production use
# training_data = DataSource(
#     name="iris_training",
#     source="file",
#     path="data/iris_train.csv",
# )

# Alternative: Database source for enterprise use
# training_data = DataSource(
#     name="iris_training",
#     source="snowflake",
#     query=Query.from_file("queries/iris_features.sql"),
# )
