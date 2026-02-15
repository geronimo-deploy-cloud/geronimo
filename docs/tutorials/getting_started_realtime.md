# Getting Started: Real-Time Endpoints

Build ML APIs with FastAPI and the Geronimo SDK.

## 1. Initialize Project

```bash
geronimo init --name my-model --template realtime
cd my-model
uv sync
source .venv/bin/activate
```

## 2. Project Structure

```
my-model/
├── geronimo.yaml
├── src/my_model/
│   ├── sdk/                      # YOUR CODE GOES HERE
│   │   ├── model.py              # Define train() and predict()
│   │   ├── features.py           # Define FeatureSet
│   │   ├── data_sources.py       # Configure data loading
│   │   ├── endpoint.py           # Define preprocess/postprocess
│   │   └── monitoring_config.py  # Latency thresholds and alerts
│   ├── app.py                    # Thin FastAPI wrapper (auto-generated)
│   ├── train.py                  # Training script
│   ├── agent.py                  # (Optional) MCP for Agentic AI Usage
│   └── monitoring/               # Metrics, alerts, drift detection
│       ├── metrics.py
│       ├── alerts.py
│       ├── drift.py
│       └── middleware.py
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

class MyModelFeatures(FeatureSet):
    """Feature engineering for my-model."""
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
from .features import MyModelFeatures
from .data_sources import training_sources


class MyModelModel(Model):
    name = "my-model"
    version = "1.0.0"

    def __init__(self):
        super().__init__()
        self.estimator: Optional[Any] = None
        self.features: Optional[MyModelFeatures] = None
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
        self.features = MyModelFeatures()
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
                                artifact_type="MyModelFeatures"))
        return paths

    def load(self, store: ArtifactStore) -> None:
        """Load trained estimator and features from ArtifactStore."""
        self.estimator = store.get("estimator")
        self.features = store.get("features")
        self._is_fitted = True
```

### Define Endpoint (`sdk/endpoint.py`)

The endpoint handles request preprocessing and response formatting:

```python
import pandas as pd
from geronimo.serving import Endpoint
from .model import MyModelModel


class MyModelEndpoint(Endpoint):
    """REST API endpoint for predictions."""

    model_class = MyModelModel

    def preprocess(self, request: dict):
        """Transform incoming request to model input."""
        # Handle both flat and nested request formats
        if "features" in request:
            req = request["features"]
        else:
            req = request

        df = pd.DataFrame([req])
        return df

    def postprocess(self, prediction):
        """Format model output for response."""
        if hasattr(prediction, "tolist"):
            prediction = prediction.tolist()
        if isinstance(prediction, list) and len(prediction) == 1:
            prediction = prediction[0]
        return {"result": prediction}

    def initialize(self, project=None, version=None):
        """Initialize endpoint — loads model from ArtifactStore."""
        super().initialize(project=project, version=version)

    def handle(self, request: dict) -> dict:
        """Handle prediction request."""
        return super().handle(request)
```

### Define Training Script (`train.py`)

```python
from geronimo.artifacts import ArtifactStore
from my_model.sdk.model import MyModelModel


def main():
    print("=" * 50)
    print("Model Training")
    print("=" * 50)

    # Train model (data loading + feature fitting is encapsulated)
    print("\n1. Training model...")
    model = MyModelModel()
    metrics = model.train()
    print(f"   Training metrics: {metrics}")

    # Save artifacts
    print("\n2. Saving artifacts...")
    store = ArtifactStore(
        project="my-model",
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

## 4. Train The Model

```bash
uv run python -m my_model.train

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

## 5. Run Locally

```bash
# Start the API server
uv run uvicorn my_model.app:app --reload
```

The `app.py` wrapper integrates your SDK endpoint with FastAPI, monitoring middleware, and optional MCP agent support. It reads configuration from `geronimo.yaml`.

## 6. Test Endpoints

```bash
# Health check
curl http://localhost:8000/health
# {"status": "ok", "mcp_enabled": true}

# View metrics
curl http://localhost:8000/metrics
# {"latency_p50_ms": ..., "latency_p99_ms": ..., "request_count": ..., "error_count": ...}

# Prediction
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": {"age": 30, "income": 75000}}'
```

## 7. Configure Monitoring (`sdk/monitoring_config.py`)

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

## 8. Deploy

```bash
geronimo generate all
# Creates: infrastructure/, Dockerfile, azure-pipelines.yaml
```

## Next Steps

- [Batch Jobs](getting_started_batch.md) — Pipeline workflows
- [Monitoring](monitoring.md) — Drift detection
- [MCP Integration](mcp_integration.md) — AI agent exposure
- [SDK Reference](sdk_reference.md) — Full API docs
