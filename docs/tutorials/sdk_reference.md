# SDK Reference

Complete API reference for the Geronimo SDK modules.

## Project Structure

```
my-project/
├── src/my_project/
│   ├── sdk/                      # YOUR CODE GOES HERE
│   │   ├── model.py              # Model class
│   │   ├── features.py           # FeatureSet class
│   │   ├── data_sources.py       # DataSource configs
│   │   ├── endpoint.py           # [realtime] Endpoint class
│   │   ├── pipeline.py           # [batch] BatchPipeline class
│   │   └── monitoring_config.py  # Thresholds and alerts
│   ├── app.py                    # [realtime] FastAPI wrapper
│   ├── agent.py                  # [realtime] MCP agent
│   ├── flow.py                   # [batch] Metaflow wrapper
│   ├── train.py                  # Training script
│   └── monitoring/               # Metrics, alerts, drift
```

---

## geronimo.data_sources

### DataSource

Supports multiple source types: `"file"`, `"function"`, and database queries.

```python
import sys
from geronimo.data_sources import DataSource, Query, collect_data_sources

# File source
training_data = DataSource(
    name="training",
    source="file",
    path="data/train.csv",
)

# Function source — load from custom code
training_data = DataSource(
    name="training",
    source="function",
    handle=my_loader_function,
)

# SQL database source
source = DataSource(
    name="training_data",
    source="snowflake",  # "postgres", "sqlserver", "file"
    query=Query.from_file("queries/train.sql"),
    connection_params={"warehouse": "ML_WH"},
)

# Load data
df = source.load(start_date="2024-01-01")

# Auto-collect by naming convention
training_sources = collect_data_sources(sys.modules[__name__], "training_")
```

### Query

```python
from geronimo.data_sources import Query

# From file
query = Query.from_file("queries/features.sql")

# Inline
query = Query("SELECT * FROM features WHERE date >= :start_date")

# Render with parameters
sql = query.render(start_date="2024-01-01")
```

---

## geronimo.features

### FeatureSet

```python
from geronimo.features import FeatureSet, Feature
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class ProjectFeatures(FeatureSet):
    # Optional: link to data source
    data_source = training_data

    # Define features
    age = Feature(dtype="numeric", transformer=StandardScaler())
    income = Feature(dtype="numeric", transformer=StandardScaler())
    segment = Feature(dtype="categorical", encoder=OneHotEncoder())
    name = Feature(dtype="text", drop=True)  # Excluded from output

# Training
features = ProjectFeatures()
X = features.fit_transform(train_df)

# Production
features.load(artifact_store)
X = features.transform(prod_df)
```

### Feature

| Parameter | Type | Description |
|-----------|------|-------------|
| `dtype` | str | `"numeric"`, `"categorical"`, `"text"` |
| `transformer` | object | sklearn transformer for numeric |
| `encoder` | object | sklearn encoder for categorical |
| `source_column` | str | Original column name if different |
| `drop` | bool | Exclude from output features |

---

## geronimo.models

### Model

The model encapsulates data loading, feature fitting, training, and persistence:

```python
from typing import Any, Optional
import numpy as np
import pandas as pd

from geronimo.models import Model, HyperParams
from geronimo.artifacts import ArtifactStore
from .features import ProjectFeatures
from .data_sources import training_sources


class ProjectModel(Model):
    name = "my-model"
    version = "1.0.0"

    def __init__(self):
        super().__init__()
        self.estimator: Optional[Any] = None
        self.features: Optional[ProjectFeatures] = None
        self._is_fitted = False

    def train(self) -> dict:
        """Self-contained training: loads data, fits features, trains estimator."""
        df = training_sources[0].load()
        for source in training_sources[1:]:
            source_df = source.load()
            if source.join_spec:
                df = df.merge(source_df,
                    left_on=source.join_spec.left_on,
                    right_on=source.join_spec.right_on,
                    how=source.join_spec.how)

        y = df["target"].values
        self.features = ProjectFeatures()
        X = self.features.fit_transform(df)

        from sklearn.ensemble import RandomForestClassifier
        params = HyperParams(n_estimators=100, max_depth=5, random_state=42)
        self.estimator = RandomForestClassifier(**params.to_dict())
        self.estimator.fit(X, y)
        self._is_fitted = True

        return {"accuracy": self.estimator.score(X, y), "n_samples": len(y)}

    def predict(self, X, return_probabilities: bool = False):
        """Transform input and generate predictions."""
        X_transformed = self.features.transform(X)
        if return_probabilities:
            return self.estimator.predict_proba(X_transformed)
        return self.estimator.predict(X_transformed)

    def save(self, store: ArtifactStore) -> list[str]:
        """Save estimator and features to ArtifactStore."""
        paths = []
        paths.append(store.save("estimator", self.estimator,
                                artifact_type=type(self.estimator).__name__))
        paths.append(store.save("features", self.features,
                                artifact_type="ProjectFeatures"))
        return paths

    def load(self, store: ArtifactStore) -> None:
        """Restore estimator and features from ArtifactStore."""
        self.estimator = store.get("estimator")
        self.features = store.get("features")
        self._is_fitted = True

# Usage
model = ProjectModel()
metrics = model.train()           # Self-contained training
store = ArtifactStore(project="my-model", version="1.0.0")
model.save(store)                  # Persist to artifact store
```

### HyperParams

```python
from geronimo.models import HyperParams

# Fixed values
params = HyperParams(n_estimators=100, max_depth=5)

# Grid search
params = HyperParams(
    n_estimators=[100, 200, 500],
    max_depth=[3, 5, 7],
)
for combo in params.grid():
    model.train(X, y, combo)
```

---

## geronimo.serving

### Endpoint

```python
import pandas as pd
from geronimo.serving import Endpoint
from .model import ProjectModel


class PredictEndpoint(Endpoint):
    model_class = ProjectModel

    def preprocess(self, request: dict):
        """Transform request to model input."""
        if "features" in request:
            req = request["features"]
        else:
            req = request
        return pd.DataFrame([req])

    def postprocess(self, prediction):
        """Format model output as response."""
        if hasattr(prediction, "tolist"):
            prediction = prediction.tolist()
        if isinstance(prediction, list) and len(prediction) == 1:
            prediction = prediction[0]
        return {"result": prediction}

    def initialize(self, project=None, version=None):
        """Load model from ArtifactStore."""
        super().initialize(project=project, version=version)

    def handle(self, request: dict) -> dict:
        """Handle prediction request."""
        return super().handle(request)
```

### app.py Wrapper

The generated `app.py` integrates your endpoint with FastAPI, monitoring, and MCP:

```python
from my_model.sdk.endpoint import PredictEndpoint
from my_model.monitoring.middleware import MonitoringMiddleware
from my_model.monitoring.metrics import MetricsCollector

app = FastAPI(title="my-model", lifespan=lifespan)
app.add_middleware(MonitoringMiddleware, collector=metrics)

# MCP agent integration (reads mcp_enabled from geronimo.yaml)
if ENABLE_MCP:
    from my_model.agent import mcp
    app.mount("/mcp", mcp.http_app())

@app.post("/predict")
def predict(request: PredictRequest):
    endpoint = get_endpoint()
    return endpoint.handle(request.model_dump())
```

---

## geronimo.batch

### BatchPipeline

```python
from geronimo.batch import BatchPipeline
from geronimo.batch.schedule import Schedule
from .model import ProjectModel
from .data_sources import training_data
from .monitoring_config import create_drift_detector, check_drift


class ScoringPipeline(BatchPipeline):
    model_class = ProjectModel
    schedule = Schedule.daily(hour=6)

    def run(self):
        """Execute batch scoring logic."""
        df = training_data.load()
        X = self.model.features.transform(df)

        # Check for drift
        detector = create_drift_detector(reference_data=df)
        drift_result = check_drift(detector, df)
        if drift_result["has_drift"]:
            print(f"⚠ Drift detected")

        predictions = self.model.predict(X)
        results = df.copy()
        results["prediction"] = predictions
        return results
```

### flow.py Wrapper

The generated `flow.py` is a thin Metaflow wrapper:

```python
from metaflow import FlowSpec, step, schedule
from my_pipeline.sdk.pipeline import ScoringPipeline


@schedule(daily=True)
class ScoringFlow(FlowSpec):
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
```

### Schedule & Trigger

```python
from geronimo.batch import Schedule, Trigger

Schedule.cron("0 6 * * *")
Schedule.daily(hour=6)
Schedule.weekly(day=0, hour=0)

Trigger.s3_upload(bucket="data", prefix="input/")
Trigger.sns_message(topic_arn="arn:aws:sns:...")
Trigger.manual()
```

---

## geronimo.artifacts

### ArtifactStore

```python
from geronimo.artifacts import ArtifactStore

# Local storage
store = ArtifactStore(project="my-model", version="1.0.0", backend="local")

# S3 storage
store = ArtifactStore(project="my-model", version="1.0.0", backend="s3", s3_bucket="ml-artifacts")

# Save
store.save("model", model.estimator)
store.save("encoder", features.encoder)

# Load
store = ArtifactStore.load(project="my-model", version="1.0.0")
model = store.get("model")

# List artifacts
store.list()  # [ArtifactMetadata(...), ...]
```

---

## SDK Monitoring Config

### Real-Time (`sdk/monitoring_config.py`)

```python
LATENCY_P50_WARNING = 100.0   # ms
LATENCY_P99_WARNING = 500.0   # ms
ERROR_RATE_WARNING = 1.0      # %
ERROR_RATE_CRITICAL = 5.0     # %

def create_alert_manager() -> AlertManager:
    """Configured with SLACK_WEBHOOK_URL env var."""
    ...

def check_thresholds(metrics, alerts) -> None:
    """Check metrics and send alerts if breached."""
    ...
```

### Batch (`sdk/monitoring_config.py`)

```python
FEATURE_DRIFT_THRESHOLD = 0.3
DATASET_DRIFT_THRESHOLD = 0.1

def create_drift_detector(reference_data):
    """Create detector for drift checking."""
    ...

def check_drift(detector, current_data, alert_manager=None):
    """Check for drift and optionally alert."""
    ...

def send_pipeline_completion_alert(alerts, result, success=True):
    """Notify on pipeline completion."""
    ...
```
