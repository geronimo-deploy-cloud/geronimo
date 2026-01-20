# Getting Started: Real-Time Endpoints

Build ML APIs with FastAPI and the Geronimo SDK.

## 1. Initialize Project

```bash
geronimo init --name my-model --template realtime
cd my-model
uv sync
```

## 2. Project Structure

```
my-model/
├── geronimo.yaml
├── src/my_model/
│   ├── sdk/                    # YOUR CODE GOES HERE
│   │   ├── model.py            # Define train() and predict()
│   │   ├── features.py         # Define FeatureSet
│   │   ├── data_sources.py     # Configure data loading
│   │   ├── endpoint.py         # Define preprocess/postprocess
│   │   └── monitoring_config.py # Thresholds and alerts
│   ├── app.py                  # Thin FastAPI wrapper (auto-generated)
│   ├── train.py                # Training script
│   └── monitoring/             # Metrics, alerts, drift detection
├── models/                     # Saved model artifacts
└── tests/
```

## 3. Implement SDK Components

### Define Features (`sdk/features.py`)

```python
from geronimo.features import FeatureSet, Feature
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class ProjectFeatures(FeatureSet):
    """Define your feature transformations."""
    # age = Feature(dtype="numeric", transformer=StandardScaler())
    # income = Feature(dtype="numeric", transformer=StandardScaler())
    # segment = Feature(dtype="categorical", encoder=OneHotEncoder())
    pass
```

### Define Model (`sdk/model.py`)

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
```

### Define Endpoint (`sdk/endpoint.py`)

```python
from geronimo.serving import Endpoint
import pandas as pd
from .model import ProjectModel

class PredictEndpoint(Endpoint):
    model_class = ProjectModel

    def preprocess(self, request: dict):
        """Transform request to model input."""
        df = pd.DataFrame([request["features"]])
        return self.model.features.transform(df)

    def postprocess(self, prediction):
        """Format model output as response."""
        return {"score": float(prediction[0][1])}
```

## 4. Run Locally

```bash
# Start the API server
uvicorn my_model.app:app --reload
```

The thin `app.py` wrapper handles FastAPI setup, imports your SDK endpoint, and adds monitoring middleware.

## 5. Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# View metrics
curl http://localhost:8000/metrics

# Prediction
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": {"age": 30, "income": 75000}}'
```

## 6. Configure Monitoring (`sdk/monitoring_config.py`)

```python
# Latency thresholds (milliseconds)
LATENCY_P50_WARNING = 100.0
LATENCY_P99_WARNING = 500.0

# Error rate thresholds (percentage)
ERROR_RATE_WARNING = 1.0
ERROR_RATE_CRITICAL = 5.0

# Enable Slack alerts
# export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
```

## 7. Deploy

```bash
geronimo generate all
# Creates: infrastructure/, Dockerfile, azure-pipelines.yaml
```

## Next Steps

- [Batch Jobs](getting_started_batch.md) — Pipeline workflows
- [Monitoring](monitoring.md) — Drift detection
- [SDK Reference](sdk_reference.md) — Full API docs
