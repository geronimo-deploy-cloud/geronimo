# Getting Started

Geronimo is an ML development framework. Choose your path:

## Installation

```bash
pip install geronimo

# With optional integrations
pip install geronimo[mlflow]      # MLflow support
pip install geronimo[databases]   # Snowflake, Postgres, SQL Server
pip install geronimo[all]         # Everything
```

## Choose Your Template

| Guide | Use Case | Command |
|-------|----------|---------|
| [Real-Time Endpoints](getting_started_realtime.md) | REST APIs, low-latency predictions | `geronimo init -t realtime` |
| [Batch Pipelines](getting_started_batch.md) | Scheduled jobs, bulk scoring | `geronimo init -t batch` |

For projects needing both:
```bash
geronimo init --name my-model --template both
```


## Documentation

- [SDK Reference](sdk_reference.md) — Full API documentation
- [Monitoring](monitoring.md) — Drift detection
- [MCP Integration](mcp_integration.md) — AI agent exposure
