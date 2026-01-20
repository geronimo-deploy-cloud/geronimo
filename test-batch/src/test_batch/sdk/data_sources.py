"""Data source definitions - configure where your data comes from.

This module is imported by model.py and pipeline.py to load training/scoring data.
"""

from geronimo.data import DataSource, Query


# =============================================================================
# Training Data Source (used by model.py)
# =============================================================================

# TODO: Configure your training data source
# Option 1: Local CSV file
training_data = DataSource(
    name="training",
    source="file",
    path="data/train.csv",  # Update with your path
)

# Option 2: Snowflake query
# training_data = DataSource(
#     name="training",
#     source="snowflake",
#     query=Query.from_file("queries/train.sql"),
# )

# Option 3: S3 parquet
# training_data = DataSource(
#     name="training",
#     source="file",
#     path="s3://my-bucket/data/train.parquet",
# )


# =============================================================================
# Scoring Data Source (used by pipeline.py for batch scoring)
# =============================================================================

scoring_data = DataSource(
    name="scoring",
    source="file",
    path="batch/data/input.csv",  # Update with your path
)
