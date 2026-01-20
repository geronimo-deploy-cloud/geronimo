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
│   ├── flow.py                   # [batch] Metaflow wrapper
│   └── train.py                  # Training script
```

---

## geronimo.data

### DataSource

```python
from geronimo.data import DataSource, Query

# File source
training_data = DataSource(
    name="training",
    source="file",
    path="data/train.csv",
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
```

### Query

```python
from geronimo.data import Query

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

```python
from geronimo.models import Model, HyperParams
from .features import ProjectFeatures
from .data_sources import training_data

class ProjectModel(Model):
    name = "my-model"
    version = "1.0.0"
    features = ProjectFeatures()
    data_source = training_data

    def train(self, X, y, params: HyperParams):
        from sklearn.ensemble import RandomForestClassifier
        self.estimator = RandomForestClassifier(**params.to_dict())
        self.estimator.fit(X, y)

    def predict(self, X):
        return self.estimator.predict_proba(X)

# Usage
model = ProjectModel()
model.train(X, y, HyperParams(n_estimators=100, max_depth=5))
model.save(store)
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
from geronimo.serving import Endpoint
from .model import ProjectModel

class PredictEndpoint(Endpoint):
    model_class = ProjectModel

    def preprocess(self, request: dict):
        """Transform request to model input."""
        import pandas as pd
        df = pd.DataFrame([request["features"]])
        return self.model.features.transform(df)

    def postprocess(self, prediction):
        """Format model output as response."""
        return {"score": float(prediction[0])}

# Usage (handled by app.py wrapper)
endpoint = PredictEndpoint()
endpoint.load()
result = endpoint.handle({"features": {"age": 30, "income": 50000}})
```

### app.py Wrapper

The generated `app.py` is a thin FastAPI wrapper (~50 lines):

```python
from my_model.sdk.endpoint import PredictEndpoint

app = FastAPI(title="my-model")
app.add_middleware(MonitoringMiddleware, collector=metrics)

@app.post("/predict")
def predict(request: PredictRequest):
    endpoint = get_endpoint()
    return endpoint.handle(request.model_dump())
```

---

## geronimo.batch

### BatchPipeline

```python
from geronimo.batch import BatchPipeline, Schedule
from .model import ProjectModel
from .data_sources import scoring_data

class ScoringPipeline(BatchPipeline):
    model_class = ProjectModel
    data_source = scoring_data
    schedule = Schedule.daily(hour=6)

    def run(self):
        """Execute batch scoring logic."""
        data = self.data_source.load()
        X = self.model.features.transform(data)
        predictions = self.model.predict(X)
        self.save_results(predictions, "batch/output/scores.parquet")
        return {"samples_scored": len(predictions)}

# Usage (handled by flow.py wrapper)
pipeline = ScoringPipeline()
pipeline.initialize()
result = pipeline.execute()
```

### flow.py Wrapper

The generated `flow.py` is a thin Metaflow wrapper (~40 lines):

```python
from my_pipeline.sdk.pipeline import ScoringPipeline

@schedule(daily=True)
class ScoringFlow(FlowSpec):
    @step
    def run_pipeline(self):
        self.pipeline = ScoringPipeline()
        self.result = self.pipeline.execute()
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
