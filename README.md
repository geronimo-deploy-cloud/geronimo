# Geronimo: The Declarative ML Framework For AI

Build, train, and deploy ML models with production-ready infrastructure and Generative AI MCP support from the start.

Geronimo is like **dbt for AI**:

## Why Geronimo?

### 🚀 Ship Models Faster

Stop writing boilerplate. One command creates a runnable project with FastAPI endpoints, monitoring, and CI/CD ready to go.

```bash
geronimo init --name fraud-detector
cd fraud-detector && uv sync
uvicorn fraud_detector.app:app --reload  # API running in seconds
```

### 🧩 Simpler Development

Define your model's **what**, not the **how**. The SDK handles preprocessing, artifact management, and deployment wiring.

```python
class FraudModel(Model):
    name = "fraud-detector"
    features = TransactionFeatures()

    def train(self, X, y, params):
        self.estimator = XGBClassifier(**params.to_dict())
        self.estimator.fit(X, y)

    def predict(self, X):
        return self.estimator.predict_proba(X)
```

### 🤖 GenAI Agent-Ready

Every project is automatically exposed as an [MCP tool](https://modelcontextprotocol.io). AI agents like Claude can call your models directly—no extra work required.

```json
{
  "mcpServers": {
    "fraud-detector": {
      "command": "uv",
      "args": ["run", "python", "-m", "fraud_detector.agent.server"]
    }
  }
}
```

> "Analyze this transaction for fraud risk"  
> → Claude calls your model → Returns risk score

---

## Getting Started

```bash
pip install geronimo
geronimo init --name my-model --template realtime
```

Choose your template:

| Template | Use Case | Output |
|----------|----------|--------|
| `realtime` | REST APIs, low-latency | FastAPI + monitoring |
| `batch` | Scheduled jobs, bulk scoring | Metaflow + drift detection |
| `both` | APIs + scheduled pipelines | Everything |

## What You Get

A complete, runnable project structure:

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

## Deploy to Production

```bash
geronimo generate all
```

Generates:
- **Terraform** — ECS Fargate infrastructure
- **Dockerfile** — Optimized for ML serving
- **CI/CD** — Azure DevOps / GitHub Actions pipelines

## Integrations

| Integration | Purpose |
|-------------|---------|
| **MLflow** | Experiment tracking, artifact store |
| **Snowflake/Postgres** | Data sources for training |
| **CloudWatch** | Production metrics |
| **Slack** | Alerts for drift/errors |
| **MCP** | AI agent tool exposure |

## Documentation

- [Getting Started: Realtime](docs/tutorials/getting_started_realtime.md)
- [Getting Started: Batch](docs/tutorials/getting_started_batch.md)
- [Monitoring & Drift Detection](docs/tutorials/monitoring.md)
- [MCP Integration](docs/tutorials/mcp_integration.md)
- [SDK Reference](docs/tutorials/sdk_reference.md)

## Installation

```bash
pip install geronimo                  # Core
pip install geronimo[mlflow]          # + MLflow
pip install geronimo[databases]       # + Snowflake, Postgres
pip install geronimo[all]             # Everything
```

---

**Apache 2.0 License** • [GitHub](https://github.com/geronimo-deploy-cloud/geronimo)
