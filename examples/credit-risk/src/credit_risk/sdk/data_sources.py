"""Data source definitions for credit risk model."""

from geronimo.data import DataSource, Query


# Training data source
training_data = DataSource(
    name="training",
    source="file",
    path="data/credit_train.csv",
)

# Scoring data source (for batch predictions)
scoring_data = DataSource(
    name="scoring",
    source="file",
    path="batch/data/applications.csv",
)
