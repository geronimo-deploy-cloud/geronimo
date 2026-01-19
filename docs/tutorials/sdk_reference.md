# SDK Reference

Complete API reference for the Geronimo SDK modules.

## geronimo.data

### DataSource

```python
from geronimo.data import DataSource, Query

# SQL database source
source = DataSource(
    name="training_data",
    source="snowflake",  # "postgres", "sqlserver", "file"
    query=Query.from_file("queries/train.sql"),
    connection_params={"warehouse": "ML_WH"},  # Optional overrides
)
df = source.load(start_date="2024-01-01")

# File source
source = DataSource(name="local", source="file", path="data/train.csv")
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

class CustomerFeatures(FeatureSet):
    # Optional: link to data source
    data_source = training_data

    # Define features
    age = Feature(dtype="numeric", transformer=StandardScaler())
    income = Feature(dtype="numeric", transformer=StandardScaler())
    segment = Feature(dtype="categorical", encoder=OneHotEncoder())
    name = Feature(dtype="text", drop=True)  # Excluded from output

# Training
features = CustomerFeatures()
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

class CreditRiskModel(Model):
    name = "credit-risk"
    version = "1.2.0"
    features = CustomerFeatures()

    def train(self, X, y, params: HyperParams):
        self.estimator = XGBClassifier(**params.to_dict())
        self.estimator.fit(X, y)

    def predict(self, X):
        return self.estimator.predict_proba(X)

# Usage
model = CreditRiskModel()
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

### MLflowArtifactStore

```python
from geronimo.artifacts import MLflowArtifactStore

store = MLflowArtifactStore(
    project="credit-risk",
    version="1.2.0",
    tracking_uri="http://localhost:5000",
)
store.save("model", model)
store.log_metrics({"auc": 0.95, "precision": 0.88})
store.log_params({"n_estimators": 100})
store.end_run()
```

---

## geronimo.serving

### Endpoint

```python
from geronimo.serving import Endpoint

class PredictEndpoint(Endpoint):
    model_class = CreditRiskModel

    def preprocess(self, request: dict):
        import pandas as pd
        df = pd.DataFrame([request["data"]])
        return self.model.features.transform(df)

    def postprocess(self, prediction):
        return {"score": float(prediction[0])}

# Usage
endpoint = PredictEndpoint()
endpoint.initialize()
result = endpoint.handle({"data": {"age": 30, "income": 50000}})
```

---

## geronimo.batch

### BatchPipeline

```python
from geronimo.batch import BatchPipeline, Schedule

class DailyScoringPipeline(BatchPipeline):
    model_class = CreditRiskModel
    schedule = Schedule.daily(hour=6)

    def run(self):
        data = self.model.features.data_source.load()
        X = self.model.features.transform(data)
        predictions = self.model.predict(X)
        self.save_results(predictions)

# Execute
pipeline = DailyScoringPipeline()
pipeline.initialize()
pipeline.execute()
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
