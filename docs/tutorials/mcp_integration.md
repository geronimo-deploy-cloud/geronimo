# AI Agent Integration (MCP)

Geronimo projects are "Agent-Ready" — expose your ML model as a tool for AI agents via the Model Context Protocol (MCP).

## Overview

Realtime projects automatically include MCP support:
- **`/mcp` endpoint** — For remote agents via Streamable HTTP
- **`agent/server.py`** — For local desktop agents (Claude Desktop) via stdio

## Quick Start

### 1. Initialize a Realtime Project

```bash
geronimo init --name my-model --template realtime
cd my-model
uv sync
```

### 2. Start the Server

```bash
uv run uvicorn my_model.app:app --reload
```

### 3. Verify MCP is Enabled

```bash
curl http://localhost:8000/health
# {"status": "ok", "mcp_enabled": true}
```

## Transports

| Transport | Use Case | How to Connect |
|-----------|----------|----------------|
| Streamable HTTP | Remote agents, web integrations | `http://localhost:8000/mcp` |
| Stdio | Local desktop agents (Claude Desktop) | `uv run python -m my_model.agent` |

## Configuration

### Enable/Disable via Environment

```bash
# Disable MCP (default: enabled)
export ENABLE_MCP_AGENT=false
uv run uvicorn my_model.app:app
```

### Configuration in geronimo.yaml

```yaml
model:
  type: realtime
  mcp_enabled: true  # default: true for realtime
```

## Using with Claude Desktop

Edit your Claude Desktop configuration:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "my-model": {
      "command": "uv",
      "args": ["run", "python", "-m", "my_model.agent"],
      "cwd": "/absolute/path/to/my-model"
    }
  }
}
```

After restarting Claude Desktop, you can use your model:

> "Use my-model to predict for a customer with income $75,000 and age 35"

## Available Tools

The generated MCP server exposes these tools:

### `predict(features: dict) -> str`

Make a prediction using your ML model.

```python
# Example usage by an AI agent
result = await predict({"age": 35, "income": 75000})
```

### `get_model_info() -> str`

Get information about the deployed model.

## How It Works

```
AI Agent → MCP Tool → SDK Endpoint → Model.predict()
               ↓             ↓
         preprocess()   postprocess()
```

The MCP tool wraps your SDK `PredictEndpoint`, which handles:
1. `preprocess()` — Transform request to model input
2. `model.predict()` — Generate prediction
3. `postprocess()` — Format response

## Customization

The agent implementation is in `src/<project>/agent/server.py`. You can add:

### Additional Tools

```python
from my_model.agent.server import mcp

@mcp.tool()
async def explain_prediction(features: dict) -> str:
    """Explain why the model made this prediction."""
    # Add SHAP or other explainability
    ...
```

### Resources

```python
@mcp.resource("model://info")
async def get_model_metadata() -> str:
    """Get model metadata."""
    return f"Model: {endpoint.model.name} v{endpoint.model.version}"
```

## Troubleshooting

### MCP dependencies not installed

```bash
pip install mcp
# or
uv add mcp
```

### MCP endpoint not available

Check if MCP is enabled:
```bash
curl http://localhost:8000/health
# Look for "mcp_enabled": true
```

If disabled, set the environment variable:
```bash
export ENABLE_MCP_AGENT=true
```
