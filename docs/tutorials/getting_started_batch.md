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
│   ├── train.py                  # Training script
│   └── monitoring/               # Metrics, alerts, drift detection
└── tests/
```

## 3. Implement SDK Components

### Define Data Source (`sdk/data_sources.py`)

DataSources support multiple source types: `"file"`, `"function"`, and database queries via `"snowflake"`, `"postgres"`, etc.

```python
import sys
from geronimo.data_sources import DataSource, Query, collect_data_sources

# File source — load from CSV/Parquet
training_data = DataSource(
    name="training",
    source="file",
    path="data/train.csv",
)

# Function source — load from custom code
# training_data = DataSource(
#     name="training",
#     source="function",
#     handle=my_loader_function,
# )

# SQL database source
# training_data = DataSource(
#     name="training",
#     source="snowflake",
#     query=Query.from_file("queries/train.sql"),
#     connection_params={"warehouse": "ML_WH"},
# )

# Auto-collect sources by naming convention (training_*, production_*)
training_sources = collect_data_sources(sys.modules[__name__], "training_")
production_sources = collect_data_sources(sys.modules[__name__], "production_")
```

### Define Features (`sdk/features.py`)

```python
from geronimo.features import FeatureSet, Feature
from sklearn.preprocessing import StandardScaler

class MyPipelineFeatures(FeatureSet):
    """Feature engineering for my-pipeline."""
    age = Feature(dtype="numeric", transformer=StandardScaler())
    income = Feature(dtype="numeric", transformer=StandardScaler())
```

### Define Model (`sdk/model.py`)

The model encapsulates data loading, feature fitting, training, and artifact persistence:

```python
from typing import Any, Optional
import numpy as np
import pandas as pd

from geronimo.models import Model, HyperParams
from geronimo.artifacts import ArtifactStore
from .features import MyPipelineFeatures
from .data_sources import training_sources


class MyPipelineModel(Model):
    name = "my-pipeline"
    version = "1.0.0"

    def __init__(self):
        super().__init__()
        self.estimator: Optional[Any] = None
        self.features: Optional[MyPipelineFeatures] = None
        self._is_fitted = False

    def train(self) -> dict:
        """Train the model.
        
        Loads training data, fits features, and trains estimator.
        """
        # Load and join training data sources
        df = training_sources[0].load()
        for source in training_sources[1:]:
            source_df = source.load()
            if source.join_spec:
                df = df.merge(
                    source_df,
                    left_on=source.join_spec.left_on,
                    right_on=source.join_spec.right_on,
                    how=source.join_spec.how,
                )

        y = df["target"].values  # TODO: your target column

        # Fit features and transform
        self.features = MyPipelineFeatures()
        X = self.features.fit_transform(df)

        # Train estimator
        from sklearn.ensemble import RandomForestClassifier
        params = HyperParams(n_estimators=100, max_depth=5, random_state=42)
        self.estimator = RandomForestClassifier(**params.to_dict())
        self.estimator.fit(X, y)
        self._is_fitted = True

        return {
            "accuracy": self.estimator.score(X, y),
            "n_samples": len(y),
            "n_features": X.shape[1],
        }

    def predict(self, X, return_probabilities: bool = False):
        """Generate predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        X_transformed = self.features.transform(X)
        if return_probabilities:
            return self.estimator.predict_proba(X_transformed)
        return self.estimator.predict(X_transformed)

    def save(self, store: ArtifactStore) -> list[str]:
        """Save trained estimator and features to ArtifactStore."""
        paths = []
        paths.append(store.save("estimator", self.estimator,
                                artifact_type=type(self.estimator).__name__))
        paths.append(store.save("features", self.features,
                                artifact_type="MyPipelineFeatures"))
        return paths

    def load(self, store: ArtifactStore) -> None:
        """Load trained estimator and features from ArtifactStore."""
        self.estimator = store.get("estimator")
        self.features = store.get("features")
        self._is_fitted = True
```

### Define Pipeline (`sdk/pipeline.py`)

```python
from geronimo.batch import BatchPipeline
from geronimo.batch.schedule import Schedule
from .model import MyPipelineModel
from .data_sources import training_data
from .monitoring_config import create_drift_detector, check_drift


class ScoringPipeline(BatchPipeline):
    model_class = MyPipelineModel
    schedule = Schedule.daily(hour=6)

    def run(self):
        """Execute batch scoring logic."""
        # Load data
        df = training_data.load()

        # Transform features using fitted model
        X = self.model.features.transform(df)

        # Check for drift
        detector = create_drift_detector(reference_data=df)
        drift_result = check_drift(detector, df)
        if drift_result["has_drift"]:
            print(f"⚠ Drift detected: {drift_result}")

        # Run predictions
        predictions = self.model.predict(X)

        # Format results
        results = df.copy()
        results["prediction"] = predictions
        return results
```

### Define Training Script (`train.py`)

```python
from geronimo.artifacts import ArtifactStore
from my_pipeline.sdk.model import MyPipelineModel


def main():
    print("=" * 50)
    print("Model Training")
    print("=" * 50)

    # Train model (data loading + feature fitting is encapsulated)
    print("\n1. Training model...")
    model = MyPipelineModel()
    metrics = model.train()
    print(f"   Training metrics: {metrics}")

    # Save artifacts
    print("\n2. Saving artifacts...")
    store = ArtifactStore(
        project="my-pipeline",
        version="1.0.0",
    )
    paths = model.save(store)
    print(f"   Saved artifacts to {len(paths)} locations")
    print(f"   Backend: {store.backend}")

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
```

### Define Flow (`flow.py`)

The flow is a thin Metaflow wrapper around your SDK pipeline:

```python
from metaflow import FlowSpec, step, schedule
from my_pipeline.sdk.pipeline import ScoringPipeline


@schedule(daily=True)
class ScoringFlow(FlowSpec):
    """Batch scoring flow — wraps SDK pipeline."""

    @step
    def start(self):
        """Initialize pipeline and load model."""
        self.pipeline = ScoringPipeline()
        self.pipeline.initialize()
        self.next(self.run_pipeline)

    @step
    def run_pipeline(self):
        """Execute the SDK pipeline."""
        self.result = self.pipeline.execute()
        self.next(self.end)

    @step
    def end(self):
        """Flow complete."""
        print(f"Pipeline complete: {self.result}")


if __name__ == "__main__":
    ScoringFlow()
```

## 4. Train The Model

```bash
uv run python -m my_pipeline.train

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
python -m my_pipeline.flow run
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

## 7. Deploy to Geronimo Deploy Cloud (Coming Soon)

### Step Functions (AWS)

```yaml
batch:
  dashboard_enabled: true
  drift_detection:
    enabled: false
    storage_bucket: model-monitoring
    sampling_rate: 0.05
  backend: step-functions
  step_functions:
    object_store_root: s3://my-bucket/metaflow
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

## 8. Configuration Reference

| Field | Description |
|-------|-------------|
| `batch.enabled` | Enable batch generation |
| `batch.backend` | `step-functions` or `airflow` |
| `batch.jobs[].flow_file` | Path to flow.py |
| `batch.jobs[].schedule` | Cron expression |
| `batch.jobs[].cpu` | CPU units |
| `batch.jobs[].memory` | Memory in MB |

## 9. Schedule Types

```python
Schedule.cron("0 6 * * *")      # Cron expression
Schedule.daily(hour=6)           # Daily at 6 AM
Schedule.weekly(day=0, hour=0)   # Sunday midnight
```

## 10. Next Steps

- [Real-Time Endpoints](getting_started_realtime.md) — API serving
- [Monitoring](monitoring.md) — Drift detection
- [SDK Reference](sdk_reference.md) — Full API docs
