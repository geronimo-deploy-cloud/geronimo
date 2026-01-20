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

## Import Existing Project

Already have ML code?

```bash
cd /path/to/existing-project
geronimo import .
```

Generates:
- `geronimo.yaml` — Deployment config
- `src/{project}/sdk/` — SDK wrappers with TODO tags
- `src/{project}/app.py` — Thin FastAPI wrapper

See `src/{project}/sdk/IMPORT_SUMMARY.md` for detected patterns and action items.

## Documentation

- [SDK Reference](sdk_reference.md) — Full API documentation
- [Monitoring](monitoring.md) — Drift detection
- [MCP Integration](mcp_integration.md) — AI agent exposure
