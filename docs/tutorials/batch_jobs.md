# Batch Jobs

Geronimo supports batch ML pipelines via Metaflow with deployment to Step Functions or Airflow.

## Overview

```
Training Data → BatchPipeline → Predictions → Output Storage
                     ↓
              ArtifactStore (loads trained model)
```

---

## Using the SDK

### Define a Pipeline

```python
from geronimo.batch import BatchPipeline, Schedule
from geronimo.artifacts import ArtifactStore

class DailyScoringPipeline(BatchPipeline):
    """Score all customers daily."""

    model_class = CreditRiskModel
    schedule = Schedule.daily(hour=6)

    def run(self):
        # Load data
        data = self.model.features.data_source.load()

        # Transform features (uses fitted encoders)
        X = self.model.features.transform(data)

        # Predict
        predictions = self.model.predict(X)

        # Save results
        self.save_results(predictions, "s3://output/daily_scores.parquet")
```

### Schedule Types

```python
from geronimo.batch import Schedule, Trigger

# Cron-based
Schedule.cron("0 6 * * *")         # Every day at 6 AM
Schedule.daily(hour=6)              # Same as above
Schedule.weekly(day=0, hour=0)      # Sunday midnight

# Event-based
Trigger.s3_upload(bucket="data", prefix="input/")
Trigger.sns_message(topic_arn="arn:aws:sns:...")
Trigger.manual()                    # CLI only
```

---

## Importing Existing Metaflow Projects

If you already have Metaflow flows:

```bash
cd /path/to/metaflow-project
geronimo import .
```

Then enable batch in `geronimo.yaml`:

```yaml
batch:
  enabled: true
  backend: step-functions  # or "airflow"

  step_functions:
    s3_root: s3://my-bucket/metaflow
    batch_queue: ml-training-queue

  jobs:
    - name: daily_training
      flow_file: flows/training_flow.py
      schedule: "0 6 * * *"
      cpu: 8
      memory: 16384
```

Generate deployment config:
```bash
geronimo generate batch
```

---

## Deployment Backends

### Step Functions (AWS)

```yaml
batch:
  backend: step-functions
  step_functions:
    s3_root: s3://my-bucket/metaflow
    batch_queue: ml-training-queue
```

Deploy:
```bash
export METAFLOW_PROFILE=production
python flows/training.py step-functions create
```

### Airflow (Astronomer)

```yaml
batch:
  backend: airflow
  airflow:
    connection_id: astronomer_default
    namespace: ml-workloads
```

Generates Airflow DAGs using `KubernetesPodOperator`.

---

## Auto-Capture Reference

Generated flows include automatic drift detection:

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

---

## Configuration Reference

| Field | Description |
|-------|-------------|
| `batch.enabled` | Enable batch generation |
| `batch.backend` | `step-functions` or `airflow` |
| `batch.jobs[].flow_file` | Path to Metaflow flow |
| `batch.jobs[].schedule` | Cron expression |
| `batch.jobs[].cpu` | CPU units |
| `batch.jobs[].memory` | Memory in MB |
