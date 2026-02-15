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
│   ├── sdk/                    # YOUR CODE GOES HERE
│   │   ├── model.py            # Define train() and predict()
│   │   ├── features.py         # Define FeatureSet
│   │   ├── data_sources.py     # Configure data loading
│   │   ├── endpoint.py         # Define preprocess/postprocess
│   │   └── monitoring_config.py # Thresholds and alerts
│   ├── app.py                  # Thin FastAPI wrapper (auto-generated)
│   ├── train.py                # Training script
│   ├── agent.py                # (Optional) MCP for Agentic AI Usage
│   └── monitoring/             # Metrics, alerts, drift detection
└── tests/
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

## 5. Run Locally

```bash
# Start the API server
uvicorn my_model.app:app --reload
```

The thin `app.py` wrapper handles FastAPI setup, imports your SDK endpoint, and adds monitoring middleware.

## 6. Test Endpoints

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

## 7. Deploy

```bash
geronimo generate all
# Creates: infrastructure/, Dockerfile, azure-pipelines.yaml
```

## Next Steps

- [Batch Jobs](getting_started_batch.md) — Pipeline workflows
- [Monitoring](monitoring.md) — Drift detection
- [SDK Reference](sdk_reference.md) — Full API docs
