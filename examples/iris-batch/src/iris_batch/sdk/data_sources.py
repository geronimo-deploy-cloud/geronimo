"""Data source definitions for iris-batch.

NAMING CONVENTIONS:
- training_* : DataSources used for model training (e.g., training_customers, training_transactions)
- production_* : DataSources used for production inference/batch scoring

JOIN BEHAVIOR:
- The FIRST DataSource in each group is the primary source
- Subsequent DataSources are joined to the primary using their join_spec
- All DataSources in a group should share a common primary key

This module is imported by model.py and pipeline.py to load training/scoring data.
"""
import sys

import pandas as pd
from sklearn.datasets import load_iris

from geronimo.data_sources import DataSource, JoinSpec, Query, collect_data_sources


# =============================================================================
# Training Data Sources
# =============================================================================

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

# =============================================================================
# Production Data Sources
# =============================================================================

# =============================================================================
# Auto-collect sources
# =============================================================================

# Automatically collect training and production sources from module
training_sources = collect_data_sources(sys.modules[__name__], "training_")
production_sources = collect_data_sources(sys.modules[__name__], "production_") 
