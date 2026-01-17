# Getting Started with Geronimo

This guide covers setting up Geronimo and creating your first ML project. You can either initialize a new project or import an existing one.

## Installation

```bash
# 1. Clone Geronimo (or install from PyPI if published)
git clone https://github.com/your-org/geronimo.git
cd geronimo

# 2. Sync dependencies
uv sync

# 3. Add to path (optional)
alias geronimo="uv run geronimo"
```

---

## Option A: Create New Project

Initialize a fresh ML serving project with best practices baked in.

### 1. Initialize Project

```bash
# Create a new project in the current directory
uv run geronimo init --name credit-risk-model
```

This creates a complete project structure:
- **FastAPI app** with `/predict` and `/health` endpoints
- **Dependencies** managed by `uv`
- **Docker** and **CI/CD** ready setup
- **MCP Agent** integration (optional)

### 2. Run Locally

```bash
cd credit-risk-model
uv sync
uv run start
```
Visit http://localhost:8000/docs to see your API.

### 3. Test the API

Once the server is running, you can test the endpoints using `curl`:

**Check Health:**
```bash
curl http://localhost:8000/health
```

**Run Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"data": {"feature1": 1.0, "feature2": 0.5}}'
```

**Test MCP Integration:**
Geronimo automatically exposes your model as an MCP tool. You can list available tools:
```bash
curl -X POST http://localhost:8000/mcp \
     -H "Content-Type: application/json" \
     -d '{"method": "list_tools", "params": {}, "jsonrpc": "2.0", "id": 1}'
```

### 4. Generate Deployment Artifacts

```bash
# Create all deployment files (Terraform, Dockerfile, Pipelines)
uv run geronimo generate all
```

---

## Option B: Import Existing Project

Import an existing UV-managed ML project into the Geronimo deployment framework.

### 1. Navigate to Project
```bash
cd /path/to/existing-project
```

### 2. Run Import Command
```bash
uv run geronimo import .
```

Geronimo will:
- Scan `pyproject.toml` for dependencies and frameworks
- Detect endpoints and model artifacts
- Generate `geronimo.yaml` configuration

### 3. Review Configuration

Edit `geronimo.yaml` to customize deployment settings:

```yaml
project:
  name: my-existing-model
  version: "1.0.0"

infrastructure:
  cpu: 1024
  memory: 4096
  scaling:
    min_instances: 2
    max_instances: 10
```

### 4. Generate Deployment Artifacts

```bash
uv run geronimo generate all
```

---

## Next Steps

- [Batch Jobs](batch_jobs.md) — Create or import batch workflows
- [Monitoring Setup](monitoring.md) — Configure drift detection
- [MCP Integration](mcp_integration.md) — Expose to AI agents
