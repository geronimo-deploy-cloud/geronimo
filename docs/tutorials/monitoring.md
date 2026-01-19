# Monitoring & Drift Detection

Geronimo provides drift detection to compare model inputs at deployment time against production data.

## Overview

```
Deploy Model → Capture Reference Snapshot
                        ↓
Production Requests → Sample Inputs → Compare to Reference
                                              ↓
                                       Drift Report + Alerts
```

---

## Configuration

Add to `geronimo.yaml`:

```yaml
monitoring:
  drift_detection:
    enabled: true
    s3_bucket: my-monitoring-bucket
    sampling_rate: 0.05       # 5% of requests
    window_days: 7            # Compare last 7 days
    drift_threshold: 0.1      # Per-feature threshold
    retention_days: 90        # Keep snapshots 90 days
```

---

## Capture Reference Baseline

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

Environment variables for databases:
- **Snowflake**: `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_ACCOUNT`, etc.
- **Postgres**: `POSTGRES_CONNECTION_STRING`
- **SQL Server**: `SQLSERVER_CONNECTION_STRING`

---

## Detect Drift

```bash
geronimo monitor detect-drift reference.json current_data.csv \
  --threshold 0.1
```

Output:
```
╭─ Drift Report ──────────────────╮
│ No significant drift            │
│                                 │
│ Features with drift: 2/15       │
│ Overall score: 13.33%           │
│ Report: drift_report.json       │
╰─────────────────────────────────╯
```

---

## Programmatic API

### Capture Reference

```python
from geronimo.monitoring.api import capture_reference_from_data

snapshot = capture_reference_from_data(
    data=training_df,
    project_name="credit-risk",
    model_version="1.2.0",
    deployment_type="realtime",
)
```

### From SQL Query

```python
from geronimo.monitoring.api import capture_reference_from_query

snapshot = capture_reference_from_query(
    sql="SELECT * FROM features WHERE date >= '2024-01-01'",
    source_system="snowflake",
    project_name="credit-risk",
)
```

---

## Auto-Capture in Batch Jobs

Generated Metaflow flows include automatic reference capture:

```python
@step
def capture_baseline(self):
    if self.capture_reference and len(self.data) > 0:
        from geronimo.monitoring.api import capture_reference_from_data

        self.reference_snapshot = capture_reference_from_data(
            data=self.data,
            project_name="my-model",
            model_version=self.model_version,
            deployment_type="batch",
        )
```

---

## Data Model

### Reference Snapshot
Captured at deploy time:
- Feature statistics (mean, std, quantiles)
- Sample of inputs
- Stored in S3 as Parquet

### Recent Window
Rolling window of production inputs for comparison.

### Drift Report
- Per-feature drift scores
- Overall dataset drift flag
- Triggered alerts

---

## Storage Architecture

```
s3://monitoring-bucket/
├── {project}/
│   ├── references/
│   │   └── {snapshot_id}.parquet
│   ├── windows/
│   │   └── {date}/{window_id}.parquet
│   └── reports/
│       └── {report_id}.json
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `geronimo monitor capture-reference` | Capture baseline snapshot |
| `geronimo monitor detect-drift` | Compare reference to current data |

### capture-reference Options

| Option | Description |
|--------|-------------|
| `--input-type` | `file` or `query` |
| `--source-system` | `snowflake`, `postgres`, `sqlserver` |
| `--sampling-rate` | Fraction to sample (0.001-1.0) |
| `--s3-bucket` | Storage bucket |
