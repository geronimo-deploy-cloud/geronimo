# Geronimo: The Declarative ML Framework For AI

Build, train, and deploy ML models with production-ready infrastructure and Generative AI MCP support from the start.

Geronimo is like **dbt for AI**:

## Why Geronimo?

### 🚀 Ship Models Faster

Stop writing boilerplate. One command creates a runnable project with FastAPI endpoints, MCP Support, monitoring, and CI/CD ready to go.

```bash
geronimo init --name iris-realtime --framework sklearn --template realtime

Initializing project: iris-realtime
  Template: realtime
╭───────────────────────────────────────────────────────────────╮
│ ✓ Project 'iris-realtime' created successfully!               │
│                                                               │
│ Template: realtime                                            │
│                                                               │
│ Next steps:                                                   │
│   1. cd iris-realtime                                         │
│   2. uv sync                                                  │
│   3. uvicorn iris_realtime.app:app --reload  # Run API server │
╰───────────────────────────────────────────────────────────────╯
```

### 🧩 Simpler Development

Define your model's **what**, not the **how**. The SDK has 4 components—each maps to one file:

| Component | File | Purpose |
|-----------|------|---------|
| **DataSource** | `data_sources.py` | Where your data comes from |
| **FeatureSet** | `features.py` | How to transform raw data |
| **Model** | `model.py` | Training and prediction logic |
| **MonitoringConfig** | `monitoring_config.py` | Monitoring and drift detection settings |

Then, depending on the template you choose, you'll also have:
| Component | File | Purpose |
|-----------|------|---------|
| **Endpoint** | `endpoint.py` | Request/response handling (realtime) |
| **Pipeline** | `pipeline.py` | Batch job orchestration (batch) |


#### Common Components:

##### data_sources.py — Declare your data
```
training_data = DataSource(name="training", source="snowflake", query=Query.from_file("train.sql"))
```

##### features.py — Define transformations
```
class IrisFeatures(FeatureSet):
    sepal_length = Feature(dtype="numeric", transformer=StandardScaler())
    sepal_width = Feature(dtype="numeric", transformer=StandardScaler())
    petal_length = Feature(dtype="numeric", transformer=StandardScaler())
    petal_width = Feature(dtype="numeric", transformer=StandardScaler())
```

##### model.py — Train and predict
```
class IrisModel(Model):
    name = "iris-realtime"
    features = IrisFeatures()
    
    def train(self, X, y, params): ...
    def predict(self, X): ...
```
##### monitoring_config.py — Configure monitoring
```
class IrisMonitoringConfig(MonitoringConfig):
    drift_threshold = 0.1
    alert_email = "[EMAIL_ADDRESS]"
```

#### Deployment Specific:

##### pipeline.py — Run batch jobs
```
class ScoringPipeline(BatchPipeline):
    schedule = Schedule.daily(hour=6)
    def run(self): ...
```

##### endpoint.py — Handle realtime requests
```
class IrisRealtimeEndpoint(Endpoint):
    def preprocess(self, request): ...
    def postprocess(self, prediction): ...
```

##### 🤖 GenAI Agent-Ready

Every project is automatically exposed as an [MCP tool](https://modelcontextprotocol.io). AI agents like Claude can call your models directly—no extra work required.

```json
{
  "mcpServers": {
    "iris-realtime": {
      "command": "uv",
      "args": ["run", "python", "-m", "iris-realtime.agent.server"]
    }
  }
}
```

> "Analyze this transaction for fraud risk"  
> → Claude calls your model → Returns risk score

---

## Getting Started

### 1. Install and Configure

```bash
pip install geronimo

# First-time setup: configure artifact storage
geronimo config init
```

The setup wizard lets you choose where to store trained models:
- **local** — `~/.geronimo/artifacts` (default)
- **s3** — Your S3 bucket
- **cloud** — Geronimo Cloud (requires `geronimo auth login`)

### 2. Create a Project

```bash
geronimo init --name my-model --template realtime
```

Choose your template:

| Template | Use Case | Output |
|----------|----------|--------|
| `realtime` | REST APIs, low-latency | FastAPI + monitoring |
| `batch` | Scheduled jobs, bulk scoring | Metaflow + drift detection |
| `both` | APIs + scheduled pipelines | Everything |

#### What You Get

A complete, runnable project structure (realtime shown):

```
my-model/
├── src/my_model/
│   ├── sdk/                    # Define your model here
│   │   ├── model.py            # train() + predict()
│   │   ├── features.py         # Feature transformations
│   │   ├── endpoint.py         # Request/response handling
│   │   └── monitoring_config.py
│   ├── app.py                  # FastAPI (auto-generated)
│   └── train.py                # Training script
├── geronimo.yaml               # Deployment config
└── models/                     # Saved artifacts
```

**Focus on the `sdk/` folder.** Everything else is generated for you.

## Integrations

| Integration | Purpose |
|-------------|---------|
| **MLflow** | Experiment tracking, artifact store |
| **Snowflake/Postgres Connectors** | Data sources for training |
| **CloudWatch** | Production metrics |
| **Slack** | Alerts for drift/errors |
| **MCP** | AI agent tool exposure |

## Deploy to Production

### Deploy Directly through Pulumi
```bash
geronimo deploy up --project my-model --target aws
```

### Deploy through CI/CD
```bash
geronimo generate all
```

This will generate:
- **Terraform** — ECS Fargate infrastructure
- **Dockerfile** — Optimized for ML serving
- **CI/CD** — Azure DevOps / GitHub Actions pipelines

### Deploy to Geronimo Cloud (managed)
```bash
geronimo auth login
geronimo deploy up --project my-model --target gdc
```

## Documentation

- [Getting Started: Realtime](docs/tutorials/getting_started_realtime.md)
- [Getting Started: Batch](docs/tutorials/getting_started_batch.md)
- [Monitoring & Drift Detection](docs/tutorials/monitoring.md)
- [MCP Integration](docs/tutorials/mcp_integration.md)
- [SDK Reference](docs/tutorials/sdk_reference.md)

## Installation Options

```bash
pip install geronimo                  # Core
pip install geronimo[mlflow]          # + MLflow
pip install geronimo[databases]       # + Snowflake, Postgres
pip install geronimo[all]             # Everything
```

---

**Apache 2.0 License** • [GitHub](https://github.com/geronimo-deploy-cloud/geronimo)
