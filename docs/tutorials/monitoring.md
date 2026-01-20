# Monitoring & Drift Detection

Geronimo provides monitoring for both **real-time** and **batch** ML deployments.

## Monitoring by Template Type

| Feature | Real-Time | Batch |
|---------|-----------|-------|
| **Latency Metrics** | ✅ p50, p99 | ❌ Not needed |
| **Error Rates** | ✅ Tracked | ❌ Not needed |
| **Drift Detection** | 📋 Future | ✅ Full support |
| **Alerts (Slack/email)** | ✅ | ✅ |

---

## Real-Time Monitoring

### Configuration (`sdk/monitoring_config.py`)

```python
# Latency thresholds (milliseconds)
LATENCY_P50_WARNING = 100.0
LATENCY_P99_WARNING = 500.0

# Error rate thresholds (percentage)
ERROR_RATE_WARNING = 1.0
ERROR_RATE_CRITICAL = 5.0
```

### Enable Slack Alerts

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

### Check Thresholds

```python
from my_model.sdk.monitoring_config import create_alert_manager, check_thresholds
from my_model.monitoring.metrics import metrics

alerts = create_alert_manager()
check_thresholds(metrics, alerts)
```

### Metrics Endpoint

Generated `app.py` includes a `/metrics` endpoint:

```bash
curl http://localhost:8000/metrics
```

Returns:
```json
{
  "latency_p50_ms": 45.2,
  "latency_p99_ms": 120.5,
  "request_count": 1000,
  "error_count": 3
}
```

---

## Batch Drift Detection

### Configuration (`sdk/monitoring_config.py`)

```python
# Drift thresholds
FEATURE_DRIFT_THRESHOLD = 0.3     # Alert if >30% of features drift
DATASET_DRIFT_THRESHOLD = 0.1     # Alert if PSI > 0.1
PREDICTION_DRIFT_THRESHOLD = 0.2  # Alert if prediction distribution shifts
```

### Usage in Pipeline

```python
from my_pipeline.sdk.monitoring_config import (
    create_drift_detector,
    check_drift,
    create_alert_manager,
    send_pipeline_completion_alert,
)

def run(self):
    alerts = create_alert_manager()
    
    # Load reference data (training sample)
    import pandas as pd
    reference = pd.read_parquet("data/training_sample.parquet")
    detector = create_drift_detector(reference_data=reference)
    
    # Load scoring data
    scoring_data = self.data_source.load()
    
    # Check for drift before scoring
    drift_result = check_drift(detector, scoring_data, alert_manager=alerts)
    if drift_result["has_drift"]:
        # Handle drift: log, pause, or continue with warning
        print(f"⚠ Drift detected: {drift_result}")
    
    # Run scoring...
    predictions = self.model.predict(X)
    
    # Send completion alert
    send_pipeline_completion_alert(alerts, {"samples_scored": len(predictions)})
```

---

## CLI Commands

### Capture Reference Baseline

```bash
# From file
geronimo monitor capture-reference data.csv \
  --project my-model \
  --input-type file \
  --sampling-rate 0.1

# From database query
geronimo monitor capture-reference query.sql \
  --project my-model \
  --input-type query \
  --source-system snowflake
```

### Detect Drift

```bash
geronimo monitor detect-drift reference.json current_data.csv \
  --threshold 0.1
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

### DriftDetector Class

```python
from my_model.monitoring.drift import DriftDetector

detector = DriftDetector(
    reference_data=training_df,
    numerical_features=["age", "income"],
    categorical_features=["segment"],
)

result = detector.calculate_drift(
    current_data=production_df,
    report_path="drift_report.html",  # Optional HTML report
)

print(result["dataset_drift"])  # True/False
print(result["drift_share"])    # % of features with drift
```

---

## Alert Manager

```python
from my_model.monitoring.alerts import AlertManager, AlertSeverity

alerts = AlertManager(cooldown_seconds=300)  # 5 min between duplicates
alerts.add_slack(webhook_url="https://hooks.slack.com/...")

# Send alert
alerts.alert(
    title="Model Performance Degraded",
    message="Error rate exceeded 5%",
    severity=AlertSeverity.CRITICAL,
    metadata={"error_rate": 5.2, "threshold": 5.0},
)

# Threshold-based alert
alerts.alert_threshold(
    metric_name="latency_p99",
    current_value=520.0,
    threshold=500.0,
    comparison="gt",  # greater than
)
```

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

## Configuration (`geronimo.yaml`)

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
