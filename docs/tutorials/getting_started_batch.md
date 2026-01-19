# Getting Started: Batch Pipelines

Build ML batch jobs with Metaflow and the Geronimo SDK.

## 1. Initialize Project

```bash
geronimo init --name my-pipeline --template batch
cd my-pipeline
uv sync
```

## 2. Project Structure

```
my-pipeline/
├── geronimo.yaml
├── src/my_pipeline/
│   ├── ml/           # Model code
│   └── sdk/          # Geronimo SDK components
│       ├── features.py
│       ├── data_sources.py
│       ├── model.py
│       └── pipeline.py
└── batch/
    └── flows/        # Generated Metaflow flows
```

## 3. Implement SDK Components

### Define Data Source (`sdk/data_sources.py`)

```python
from geronimo.data import DataSource, Query

training_data = DataSource(
    name="customers",
    source="snowflake",
    query=Query.from_file("queries/customers.sql"),
)
```

### Define Features (`sdk/features.py`)

```python
from geronimo.features import FeatureSet, Feature
from sklearn.preprocessing import StandardScaler

class CustomerFeatures(FeatureSet):
    data_source = training_data

    age = Feature(dtype="numeric", transformer=StandardScaler())
    income = Feature(dtype="numeric", transformer=StandardScaler())
```

### Define Model (`sdk/model.py`)

```python
from geronimo.models import Model, HyperParams
from .features import CustomerFeatures

class ScoringModel(Model):
    name = "scoring"
    version = "1.0.0"
    features = CustomerFeatures()

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
from .model import ScoringModel

class DailyScoringPipeline(BatchPipeline):
    model_class = ScoringModel
    schedule = Schedule.daily(hour=6)

    def run(self):
        # Load data
        data = self.model.features.data_source.load()

        # Transform features
        X = self.model.features.transform(data)

        # Predict
        predictions = self.model.predict(X)

        # Save results
        self.save_results(predictions, "s3://output/scores.parquet")
```

## 4. Run Locally

```bash
# Execute pipeline
python -c "
from my_pipeline.sdk.pipeline import DailyScoringPipeline
pipeline = DailyScoringPipeline()
pipeline.initialize()
pipeline.execute()
"
```

## 5. Deploy to Step Functions

Configure `geronimo.yaml`:

```yaml
batch:
  enabled: true
  backend: step-functions
  step_functions:
    s3_root: s3://my-bucket/metaflow
    batch_queue: ml-training-queue
  jobs:
    - name: daily_scoring
      schedule: "0 6 * * *"
```

Generate and deploy:

```bash
geronimo generate batch
python batch/flows/daily_scoring.py step-functions create
```

## 6. Schedule Types

```python
Schedule.cron("0 6 * * *")      # Cron expression
Schedule.daily(hour=6)           # Daily at 6 AM
Schedule.weekly(day=0, hour=0)   # Sunday midnight
```

## Next Steps

- [Real-Time Endpoints](getting_started_realtime.md) — API serving
- [Monitoring](monitoring.md) — Drift detection
- [SDK Reference](sdk_reference.md) — Full API docs
