# AI Agent Integration (MCP)

Geronimo projects are "Agent-Ready" — they can be used as tools by AI agents via the Model Context Protocol (MCP).

## Overview

Every generated project can expose your SDK endpoint as an MCP tool. This allows AI assistants like Claude to call your model directly.

## Transports

| Transport | Use Case | Endpoint |
|-----------|----------|----------|
| Stdio | Local desktop agents (Claude Desktop) | stdin/stdout |
| Streamable HTTP | Remote agents, web integrations | `/mcp` |

## Configuration

MCP integration is toggleable via environment variable:

```bash
# Disable MCP (default: enabled)
export ENABLE_MCP_AGENT=false
```

## Using with Claude Desktop

### 1. Build Your Project

```bash
geronimo init --name my-model --template realtime
cd my-model
uv sync
```

### 2. Configure Claude Desktop

Edit your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "my-model": {
      "command": "uv",
      "args": ["run", "python", "-m", "my_model.agent.server"],
      "cwd": "/absolute/path/to/my-model"
    }
  }
}
```

### 3. Restart Claude Desktop

After restarting, Claude can use your model:

> "Use my-model to predict for a customer with income $75,000 and age 35"

## Testing via Streamable HTTP

### 1. Start the Server

```bash
uvicorn my_model.app:app --reload
```

### 2. Verify Endpoint

```bash
# Check health and MCP endpoint
curl http://localhost:8000/health
curl http://localhost:8000/mcp
```

### 3. Connect with an MCP Client

Use any MCP-compatible client to connect to `http://localhost:8000/mcp`.

## Tool Definition

The generated tool wraps your SDK endpoint:

```python
# agent/server.py
from mcp.server.fastmcp import FastMCP
from my_model.sdk.endpoint import PredictEndpoint

mcp = FastMCP("my-model-agent")
endpoint = PredictEndpoint()
endpoint.load()

@mcp.tool()
async def predict(features: dict) -> str:
    """Make a prediction using the ML model."""
    result = endpoint.handle({"features": features})
    return str(result)
```

## How It Works with SDK

```
Claude Request → MCP Tool → SDK Endpoint → Model.predict()
                    ↓             ↓
              preprocess()   postprocess()
```

The MCP tool simply calls your SDK `PredictEndpoint`, which handles:
1. `preprocess()` — Transform request to model input
2. `model.predict()` — Generate prediction
3. `postprocess()` — Format response

## Customization

The agent implementation is in `src/<project>/agent/server.py`. You can:

- Add additional tools (e.g., `get_feature_importance`, `explain_prediction`)
- Add resources (expose model metadata)
- Add prompts (pre-built queries for common use cases)

```python
@mcp.tool()
async def explain_prediction(features: dict) -> str:
    """Explain why the model made this prediction."""
    # Add SHAP or other explainability
    ...

@mcp.resource("model://info")
async def get_model_info() -> str:
    """Get model metadata."""
    return f"Model: {endpoint.model.name} v{endpoint.model.version}"
```
