# Batch Job Import

Import existing Metaflow projects into Geronimo for deployment to Step Functions or Airflow.

## Overview

Your Metaflow flows are already written. Geronimo generates the deployment configuration to run them on:
- **AWS Step Functions** with Batch compute (current)
- **Astronomer Airflow** with K8s Pod Operators (future)

## Step 1: Import Your Metaflow Project

```bash
cd /path/to/your-metaflow-project
uv run geronimo import .
```

Geronimo detects your existing flows and generates configuration.

## Step 2: Enable Batch in Configuration

Edit `geronimo.yaml` to enable batch jobs:

```yaml
project:
  name: training-pipeline
  version: "1.0.0"

batch:
  enabled: true
  backend: step-functions  # or "airflow"
  
  step_functions:
    s3_root: s3://your-bucket/metaflow
    batch_queue: ml-training-queue
  
  jobs:
    - name: daily_training
      flow_file: flows/training_flow.py
      schedule: "0 6 * * *"
      cpu: 8
      memory: 16384
    - name: weekly_retrain
      flow_file: flows/retrain_flow.py
      schedule: "0 0 * * 0"
      cpu: 16
      memory: 32768
```

## Step 3: Generate Deployment Config

```bash
uv run geronimo generate batch
```

This creates:
```
batch/
├── step_functions_config.json   # Metaflow environment config
└── flows/                        # Deployment wrappers (if needed)
```

## Step 4: Deploy to Step Functions

```bash
# Set environment
export METAFLOW_PROFILE=production
export $(cat batch/step_functions_config.json | jq -r 'to_entries|map("\(.key)=\(.value)")|.[]')

# Create Step Functions workflow
python flows/training_flow.py step-functions create
```

## Airflow Backend

For Astronomer deployment, set `backend: airflow`:

```yaml
batch:
  backend: airflow
  airflow:
    connection_id: astronomer_default
    namespace: ml-workloads
```

This generates Airflow DAGs using `KubernetesPodOperator` to run your flows.

## Existing Flow Requirements

Your Metaflow flows should:
1. Use `@batch` decorator for compute-intensive steps
2. Be runnable via `python flow.py run`
3. Store artifacts in S3 (configured via `s3_root`)

**Example existing flow:**
```python
from metaflow import FlowSpec, step, batch

class TrainingFlow(FlowSpec):
    @step
    def start(self):
        self.next(self.train)

    @batch(cpu=8, memory=16384)
    @step
    def train(self):
        # Your training code
        self.next(self.end)

    @step
    def end(self):
        pass
```

## Configuration Reference

| Field | Description |
|-------|-------------|
| `batch.enabled` | Enable batch generation |
| `batch.backend` | `step-functions` or `airflow` |
| `batch.step_functions.s3_root` | S3 path for Metaflow data |
| `batch.step_functions.batch_queue` | AWS Batch queue name |
| `batch.airflow.namespace` | K8s namespace for pods |
| `batch.jobs[].flow_file` | Path to your existing flow |
| `batch.jobs[].schedule` | Cron expression |
