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
│   ├── api/          # FastAPI endpoints
│   ├── ml/           # Model code
│   └── sdk/          # Geronimo SDK components
│       ├── features.py
│       ├── data_sources.py
│       ├── model.py
│       └── endpoint.py
└── tests/
```

## 3. Implement SDK Components

### Define Features (`sdk/features.py`)

```python
from geronimo.features import FeatureSet, Feature
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class CustomerFeatures(FeatureSet):
    age = Feature(dtype="numeric", transformer=StandardScaler())
    income = Feature(dtype="numeric", transformer=StandardScaler())
    segment = Feature(dtype="categorical", encoder=OneHotEncoder())
```

### Define Model (`sdk/model.py`)

```python
from geronimo.models import Model, HyperParams
from .features import CustomerFeatures

class CreditRiskModel(Model):
    name = "credit-risk"
    version = "1.0.0"
    features = CustomerFeatures()

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
from .model import CreditRiskModel

class PredictEndpoint(Endpoint):
    model_class = CreditRiskModel

    def preprocess(self, request: dict):
        df = pd.DataFrame([request["data"]])
        return self.model.features.transform(df)

    def postprocess(self, prediction):
        return {"score": float(prediction[0][1]), "class": "approved" if prediction[0][1] > 0.5 else "denied"}
```

## 4. Run Locally

```bash
uv run start
```

## 5. Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"data": {"age": 30, "income": 75000, "segment": "premium"}}'
```

## 6. Deploy

```bash
geronimo generate all
# Creates: infrastructure/, Dockerfile, azure-pipelines.yaml
```

## Next Steps

- [Batch Jobs](getting_started_batch.md) — Pipeline workflows
- [Monitoring](monitoring.md) — Drift detection
- [SDK Reference](sdk_reference.md) — Full API docs
