# AI Agent Integration (MCP)

Geronimo projects are "Agent-Ready" — they can be used as tools by AI agents via the Model Context Protocol (MCP).

## Overview

Every generated project includes an `agent/` package that exposes your model as an MCP tool. This allows AI assistants like Claude to call your model directly.

## Transports

Two transport options are available:

| Transport | Use Case | Endpoint |
|-----------|----------|----------|
| Stdio | Local desktop agents (Claude Desktop) | N/A (stdin/stdout) |
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
cd examples/credit-risk
uv sync
```

### 2. Configure Claude Desktop

Edit your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "credit-risk-model": {
      "command": "uv",
      "args": ["run", "python", "-m", "credit_risk.agent.server"],
      "cwd": "/absolute/path/to/examples/credit-risk"
    }
  }
}
```

### 3. Restart Claude Desktop

After restarting, Claude can use your model:

> "Evaluate credit risk for a customer with income $75,000 and age 35"

## Testing via Streamable HTTP

### 1. Start the Server

```bash
uv run uvicorn credit_risk.api.main:app
```

### 2. Verify Endpoint

```bash
# Check MCP endpoint is available
curl http://localhost:8000/mcp
```

### 3. Connect with an MCP Client

Use any MCP-compatible client to connect to `http://localhost:8000/mcp`.

## Tool Definition

The generated tool wraps your `predict` endpoint:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("credit-risk-agent")

@mcp.tool()
async def predict(input_data: dict) -> str:
    """Make a prediction using the ML model."""
    predictor = get_predictor()
    result = predictor.predict(input_data)
    return str(result)
```

## Customization

The agent implementation is in `src/<project>/agent/server.py`. You can add additional tools, resources, or prompts as needed.
