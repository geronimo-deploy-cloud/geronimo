# Getting Started: Batch Pipelines

Build ML batch jobs with Metaflow and the Geronimo SDK.

## 1. Initialize Project

```bash
geronimo init --name my-pipeline --template batch
cd my-pipeline
uv sync
source .venv/bin/activate
```

## 2. Project Structure

```
my-pipeline/
├── geronimo.yaml
├── src/my_pipeline/
│   ├── sdk/                      # YOUR CODE GOES HERE
│   │   ├── model.py              # Define train() and predict()
│   │   ├── features.py           # Define FeatureSet
│   │   ├── data_sources.py       # Configure data loading
│   │   ├── pipeline.py           # Define run() logic
│   │   └── monitoring_config.py  # Drift thresholds and alerts
│   ├── flow.py                   # Thin Metaflow wrapper (auto-generated)
│   └── train.py                  # Training script
```

## 3. Implement SDK Components

### Define Data Source (`sdk/data_sources.py`)

```python
from geronimo.data_sources import DataSource, Query

# Training data source
training_data = DataSource(
    name="training",
    source="file",
    path="data/train.csv",
)

# Scoring data for batch predictions
scoring_data = DataSource(
    name="scoring",
    source="file",
    path="batch/data/input.csv",
)
```

### Define Features (`sdk/features.py`)

```python
from geronimo.features import FeatureSet, Feature
from sklearn.preprocessing import StandardScaler

class ProjectFeatures(FeatureSet):
    # age = Feature(dtype="numeric", transformer=StandardScaler())
    # income = Feature(dtype="numeric", transformer=StandardScaler())
    pass
```

### Define Model (`sdk/model.py`)

```python
from geronimo.models import Model, HyperParams
from .features import ProjectFeatures
from .data_sources import training_data

class ProjectModel(Model):
    name = "scoring"
    version = "1.0.0"
    features = ProjectFeatures()
    data_source = training_data

    def train(self, X, y, params: HyperParams):
        from xgboost import XGBClassifier
        self.estimator = XGBClassifier(**params.to_dict())
        self.estimator.fit(X, y)

    def predict(self, X):
        return self.estimator.predict_proba(X)
```

### Define Pipeline (`sdk/pipeline.py`)

```python
from geronimo.batch import BatchPipeline, Schedule
from .model import ProjectModel
from .data_sources import scoring_data

class ScoringPipeline(BatchPipeline):
    model_class = ProjectModel
    data_source = scoring_data
    schedule = Schedule.daily(hour=6)

    def run(self):
        """Execute batch scoring logic.
        
        This method is called by the flow.py wrapper.
        """
        # Load scoring data
        data = self.data_source.load()
        
        # Transform features
        X = self.model.features.transform(data)
        
        # Predict
        predictions = self.model.predict(X)
        
        # Save results
        self.save_results(predictions, "batch/output/scores.parquet")
        
        return {"samples_scored": len(predictions)}
```

### Define Training Script (`train.py`)

```python
from geronimo.artifacts import ArtifactStore
from .sdk.model import ProjectModel


def main():
    # Load data
    data = training_data.load()
    
    # Initialize and train model
    model = ProjectModel()
    metrics = model.train(data)
    
    # Save model
    store = ArtifactStore(
        project="my-pipeline",
        version="1.0.0",
    )
    store.save("model", model, artifact_type="ProjectModel")
    store.save("metrics", metrics, artifact_type="Metrics")
    
    return metrics
```

### Define Flow (`flow.py`)

```python
from geronimo.flow import Flow
from .sdk.pipeline import ScoringPipeline
from .sdk.model import ProjectModel


class MyPipeline(Flow):
    name = "my-pipeline"
    version = "1.0.0"
    
    def run(self):
        # Train model
        metrics = ProjectModel().train()
        
        # Score data
        pipeline = ScoringPipeline()
        pipeline.initialize()
        result = pipeline.execute()
        
        return {"metrics": metrics, "result": result}
```

## 4. Train The Model

```bash
python train.py

==================================================
Model Training
==================================================

1. Training model...
   Training metrics: {'accuracy': 1.0, 'n_samples': 150, 'n_features': 4}

2. Saving artifacts...
   Saved artifacts to 2 locations
   Backend: local

==================================================
Training complete!
==================================================
```

## 5. Run Pipeline Locally

```bash
# Execute via the flow.py wrapper
python flow.py run

Running pylint...
    Pylint not found, so extra checks are disabled.
2026-02-15 13:59:30.813 Workflow starting (run-id 1771181970812176):
2026-02-15 13:59:30.822 [1771181970812176/start/1 (pid 17442)] Task is starting.
2026-02-15 13:59:31.738 [1771181970812176/start/1 (pid 17442)] Initialized: IrisBatchScoringPipeline(model=IrisBatchModel, schedule=Schedule(0 6 * * *), initialized)
```

## 6. Configure Drift Detection (`sdk/monitoring_config.py`)

```python
# Drift thresholds
FEATURE_DRIFT_THRESHOLD = 0.3     # Alert if >30% of features drift
DATASET_DRIFT_THRESHOLD = 0.1     # Alert if PSI > 0.1
PREDICTION_DRIFT_THRESHOLD = 0.2  # Alert if prediction dist shifts

# Usage in pipeline:
from .monitoring_config import create_drift_detector, check_drift

def run(self):
    detector = create_drift_detector(reference_data=training_df)
    drift_result = check_drift(detector, scoring_df)
    if drift_result["has_drift"]:
        # Log warning or pause pipeline
        pass
```

## 6. Deploy to Geronimo Deploy Cloud (Coming Soon)

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
geronimo generate batch
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

## 7. Configuration Reference

| Field | Description |
|-------|-------------|
| `batch.enabled` | Enable batch generation |
| `batch.backend` | `step-functions` or `airflow` |
| `batch.jobs[].flow_file` | Path to flow.py |
| `batch.jobs[].schedule` | Cron expression |
| `batch.jobs[].cpu` | CPU units |
| `batch.jobs[].memory` | Memory in MB |

## 8. Schedule Types

```python
Schedule.cron("0 6 * * *")      # Cron expression
Schedule.daily(hour=6)           # Daily at 6 AM
Schedule.weekly(day=0, hour=0)   # Sunday midnight
```

## 9. Next Steps

- [Real-Time Endpoints](getting_started_realtime.md) — API serving
- [Monitoring](monitoring.md) — Drift detection
- [SDK Reference](sdk_reference.md) — Full API docs
