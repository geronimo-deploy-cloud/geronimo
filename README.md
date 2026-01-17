# Geronimo

**MLOps Deployment Platform** — Automate ML model deployments to AWS with production-ready infrastructure.

## Overview

Geronimo generates complete deployment scaffolding for your ML models including:
- FastAPI-based serving application
- Terraform infrastructure (ECS/ECR)
- Docker configuration
- CI/CD pipelines (Azure DevOps)
- Model monitoring (metrics, drift, alerts)
- AI Agent integration (MCP)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/geronimo.git
cd geronimo

# Install with uv
uv sync

# Verify installation
uv run geronimo --version
```

### Create Your First Project

```bash
# Initialize a new ML project
uv run geronimo init --name my-model --framework sklearn --output ./

# Navigate to the project
cd my-model

# Install dependencies
uv sync --extra dev

# Run the API locally
uv run uvicorn my_model.api.main:app --reload

# Run tests
uv run pytest
```

## Commands

| Command | Description |
|---------|-------------|
| `geronimo init` | Create a new ML deployment project |
| `geronimo generate terraform` | Generate Terraform infrastructure |
| `geronimo generate dockerfile` | Generate optimized Dockerfile |
| `geronimo generate pipeline` | Generate CI/CD pipeline |
| `geronimo generate all` | Generate all deployment artifacts |
| `geronimo validate` | Validate configuration |
| `geronimo import` | Import an existing project |

## Tutorials

- [Getting Started](docs/tutorials/getting_started.md) — First project walkthrough
- [Monitoring Setup](docs/tutorials/monitoring.md) — Configure metrics, drift detection, and alerts
- [AI Agent Integration](docs/tutorials/mcp_integration.md) — Expose models to AI agents via MCP

## Configuration

Projects use `geronimo.yaml` for configuration:

```yaml
project:
  name: my-model
  version: "1.0.0"

deployment:
  compute:
    cpu: 512
    memory: 1024
    min_instances: 1
    max_instances: 10
  aws:
    region: us-east-1
```

## License

MIT
