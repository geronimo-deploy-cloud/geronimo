# Drift Detection

Monitor data drift between model deploy time and production.

## Configuration

Add to `geronimo.yaml`:
```yaml
monitoring:
  drift_detection:
    enabled: true
    s3_bucket: my-monitoring-bucket
    sampling_rate: 0.05       # 5% of data
    window_days: 7            # Compare last 7 days
    drift_threshold: 0.1      # Per-feature threshold
    retention_days: 90        # Keep snapshots 90 days
```

## Capture Reference (CLI)

### From File
```bash
geronimo monitor capture-reference data.csv \
  --project my-model \
  --input-type file \
  --sampling-rate 0.1
```

### From Database Query
```bash
geronimo monitor capture-reference query.sql \
  --project my-model \
  --input-type query \
  --source-system snowflake
```

Required environment variables for query mode:
- **Snowflake**: `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_ACCOUNT`, etc.
- **Postgres**: `POSTGRES_CONNECTION_STRING`
- **SQL Server**: `SQLSERVER_CONNECTION_STRING`

## Detect Drift (CLI)

```bash
geronimo monitor detect-drift reference.json current_data.csv
```

## Auto-Capture in Metaflow

Generated flows include automatic reference capture:

```python
@step
def capture_baseline(self):
    """Capture reference snapshot for drift detection."""
    if self.capture_reference:
        from geronimo.monitoring.api import capture_reference_from_data
        
        self.reference_snapshot = capture_reference_from_data(
            data=self.data,
            project_name="my-model",
            model_version=self.model_version,
        )
```

Disable with:
```bash
python flow.py run --capture_reference=False
```

## Programmatic API

```python
from geronimo.monitoring.api import (
    capture_reference_from_data,
    capture_reference_from_query,
)

# From DataFrame
snapshot = capture_reference_from_data(
    data=df,
    project_name="my-model",
    model_version="1.2.0",
)

# From SQL
snapshot = capture_reference_from_query(
    sql="SELECT * FROM features",
    source_system="snowflake",
    project_name="my-model",
)
```
