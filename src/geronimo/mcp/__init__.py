"""Geronimo MCP (Model Context Protocol) Module.

Provides MCP server functionality for exposing ML models to AI agents.
"""

from geronimo.mcp.server import MCPServer
from geronimo.mcp.tools import Tool, ToolResult

__all__ = ["MCPServer", "Tool", "ToolResult"]
