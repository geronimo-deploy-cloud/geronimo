# Batch Jobs

Geronimo supports batch ML pipelines via Metaflow with deployment to Step Functions or Airflow.

## Overview

```
SDK Pipeline → flow.py Wrapper → Metaflow → Step Functions/Airflow
     ↓
  run() → Load Data → Transform → Predict → Save Results
                                      ↓
                              Drift Detection (optional)
```

---

## Project Structure

```
my-pipeline/
├── src/my_pipeline/
│   ├── sdk/
│   │   ├── model.py              # train() / predict()
│   │   ├── features.py           # FeatureSet
│   │   ├── data_sources.py       # DataSource configs
│   │   ├── pipeline.py           # YOUR BATCH LOGIC HERE
│   │   └── monitoring_config.py  # Drift thresholds + alerts
│   ├── flow.py                   # Metaflow wrapper (auto-generated)
│   └── train.py                  # Training script
├── batch/
│   ├── data/                     # Input data
│   └── output/                   # Results
```

---

## Using the SDK

### Define a Pipeline (`sdk/pipeline.py`)

```python
from geronimo.batch import BatchPipeline, Schedule
from .model import ProjectModel
from .data_sources import scoring_data

class ScoringPipeline(BatchPipeline):
    """Score all customers daily."""

    model_class = ProjectModel
    data_source = scoring_data
    schedule = Schedule.daily(hour=6)

    def run(self):
        # Load data from configured source
        data = self.data_source.load()

        # Transform features (uses fitted encoders)
        X = self.model.features.transform(data)

        # Predict
        predictions = self.model.predict(X)

        # Save results
        self.save_results(predictions, "batch/output/scores.parquet")
        
        return {"samples_scored": len(predictions)}
```

### Run Locally

```bash
# Via the flow.py wrapper
python -m my_pipeline.flow run

# Or directly via SDK
python -c "
from my_pipeline.sdk.pipeline import ScoringPipeline
pipeline = ScoringPipeline()
pipeline.initialize()
print(pipeline.execute())
"
```

---

## The flow.py Wrapper

Generated `flow.py` is a thin Metaflow wrapper (~40 lines):

```python
from metaflow import FlowSpec, step, schedule
from my_pipeline.sdk.pipeline import ScoringPipeline

@schedule(daily=True)
class ScoringFlow(FlowSpec):
    """Batch scoring flow - wraps SDK pipeline."""

    @step
    def start(self):
        self.pipeline = ScoringPipeline()
        self.pipeline.initialize()
        self.next(self.run_pipeline)

    @step
    def run_pipeline(self):
        self.result = self.pipeline.execute()
        self.next(self.end)

    @step
    def end(self):
        print(f"Pipeline complete: {self.result}")


if __name__ == "__main__":
    ScoringFlow()
```

---

## Schedule Types

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

## Drift Detection in Batch

Unlike realtime, batch jobs support full drift detection:

```python
# sdk/pipeline.py
from .monitoring_config import (
    create_drift_detector,
    check_drift,
    create_alert_manager,
    send_pipeline_completion_alert,
)

def run(self):
    alerts = create_alert_manager()
    
    # Load reference data
    import pandas as pd
    reference = pd.read_parquet("data/training_sample.parquet")
    detector = create_drift_detector(reference_data=reference)
    
    # Check for drift before scoring
    data = self.data_source.load()
    drift_result = check_drift(detector, data, alert_manager=alerts)
    
    if drift_result["has_drift"]:
        print(f"⚠ Data drift detected: {drift_result['drift_result']['drift_share']*100:.1f}%")
    
    # Continue with scoring...
    predictions = self.model.predict(X)
    
    # Send completion notification
    send_pipeline_completion_alert(alerts, {"samples_scored": len(predictions)})
```

---

## Deployment Backends

### Step Functions (AWS)

```yaml
batch:
  enabled: true
  backend: step-functions
  step_functions:
    s3_root: s3://my-bucket/metaflow
    batch_queue: ml-training-queue
```

Deploy:
```bash
export METAFLOW_PROFILE=production
python -m my_pipeline.flow step-functions create
```

### Airflow (Astronomer)

```yaml
batch:
  enabled: true
  backend: airflow
  airflow:
    connection_id: astronomer_default
    namespace: ml-workloads
```

Generates Airflow DAGs using `KubernetesPodOperator`.

---

## Configuration (`geronimo.yaml`)

```yaml
batch:
  enabled: true
  backend: step-functions  # or "airflow"

  step_functions:
    s3_root: s3://my-bucket/metaflow
    batch_queue: ml-training-queue

  jobs:
    - name: daily_scoring
      flow_file: src/my_pipeline/flow.py
      schedule: "0 6 * * *"
      cpu: 8
      memory: 16384
```

Generate deployment config:
```bash
geronimo generate batch
```

---

## Configuration Reference

| Field | Description |
|-------|-------------|
| `batch.enabled` | Enable batch generation |
| `batch.backend` | `step-functions` or `airflow` |
| `batch.jobs[].flow_file` | Path to flow.py |
| `batch.jobs[].schedule` | Cron expression |
| `batch.jobs[].cpu` | CPU units |
| `batch.jobs[].memory` | Memory in MB |
