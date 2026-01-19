"""Data source definitions."""

from geronimo.data import DataSource, Query

# Example query-based source:
# training_data = DataSource(
#     name="training",
#     source="snowflake",
#     query=Query.from_file("queries/train.sql"),
# )

# Example file-based source:
# local_data = DataSource(name="local", source="file", path="data/train.csv")
