# Geronimo

**ML Development Framework** — Build, train, and deploy ML models with production-ready infrastructure.

Geronimo is like **dbt for ML**: a framework developers build around, not just deployment scaffolding.

## Features

| Category | Capabilities |
|----------|--------------|
| **SDK** | DataSource, FeatureSet, Model, Endpoint, BatchPipeline |
| **Artifacts** | Versioned storage for models, encoders, transformers (local/S3/MLflow) |
| **Deployment** | Terraform, Docker, CI/CD pipelines, real-time + batch |
| **Monitoring** | Metrics, drift detection, alerting |
| **Integrations** | MLflow, Snowflake, Postgres, SQL Server, MCP |

## Installation

```bash
# Core installation
pip install geronimo

# With optional integrations
pip install geronimo[mlflow]      # MLflow artifact store
pip install geronimo[databases]   # Snowflake, Postgres, SQL Server
pip install geronimo[all]         # Everything
```

Or from source:
```bash
git clone https://github.com/your-org/geronimo.git
cd geronimo && uv sync
```

## Quick Start

### Option A: New Project

```bash
# Initialize a new ML project
geronimo init --name credit-risk-model

cd credit-risk-model
uv sync
uv run start
```

### Option B: Import Existing Project

```bash
cd /path/to/existing-project
geronimo import .
```

This generates:
- `geronimo.yaml` — Deployment configuration
- `geronimo_sdk/` — SDK wrappers with TODO tags for manual config

## SDK Overview

### Data Layer

```python
from geronimo.data import DataSource, Query

training_data = DataSource(
    name="features",
    source="snowflake",
    query=Query.from_file("queries/training.sql"),
)

df = training_data.load(start_date="2024-01-01")
```

### Feature Engineering

```python
from geronimo.features import FeatureSet, Feature
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class CustomerFeatures(FeatureSet):
    age = Feature(dtype="numeric", transformer=StandardScaler())
    segment = Feature(dtype="categorical", encoder=OneHotEncoder())

# Training: fit + transform
X = features.fit_transform(train_df)

# Production: transform only (uses fitted encoders)
X = features.transform(prod_df)
```

### Model Definition

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
```

### Artifact Storage

```python
from geronimo.artifacts import ArtifactStore

# Save during training
store = ArtifactStore(project="credit-risk", version="1.2.0")
store.save("model", model.estimator)
store.save("encoder", features.encoder)

# Load in production
store = ArtifactStore.load(project="credit-risk", version="1.2.0")
model = store.get("model")
```

With MLflow (`pip install geronimo[mlflow]`):
```python
from geronimo.artifacts import MLflowArtifactStore

store = MLflowArtifactStore(project="credit-risk", version="1.2.0")
store.save("model", model)
store.log_metrics({"auc": 0.95})
```

### Production Endpoints

```python
from geronimo.serving import Endpoint

class PredictEndpoint(Endpoint):
    model_class = CreditRiskModel

    def preprocess(self, request):
        return self.model.features.transform(request["data"])

    def postprocess(self, prediction):
        return {"score": float(prediction[0])}
```

### Batch Pipelines

```python
from geronimo.batch import BatchPipeline, Schedule

class DailyScoringPipeline(BatchPipeline):
    model_class = CreditRiskModel
    schedule = Schedule.daily(hour=6)

    def run(self):
        data = self.model.features.data_source.load()
        predictions = self.model.predict(data)
        self.save_results(predictions)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `geronimo init` | Create new ML project |
| `geronimo import` | Import existing project with SDK wrappers |
| `geronimo generate all` | Generate Terraform, Docker, CI/CD |
| `geronimo generate batch` | Generate Metaflow/Airflow pipelines |
| `geronimo validate` | Validate configuration |
| `geronimo monitor capture-reference` | Capture baseline for drift detection |
| `geronimo monitor detect-drift` | Compare current data to baseline |

## Configuration

```yaml
# geronimo.yaml
project:
  name: credit-risk-model
  version: "1.2.0"

artifacts:
  store: s3://ml-artifacts/

batch:
  enabled: true
  backend: step-functions
  jobs:
    - name: daily_scoring
      schedule: "0 6 * * *"

monitoring:
  drift_detection:
    enabled: true
    sampling_rate: 0.05
    window_days: 7
```

## Documentation

- [Getting Started](docs/tutorials/getting_started.md)
- [Batch Jobs](docs/tutorials/batch_jobs.md)
- [Drift Detection](docs/tutorials/monitoring.md)
- [MCP Integration](docs/tutorials/mcp_integration.md)

## License

MIT
