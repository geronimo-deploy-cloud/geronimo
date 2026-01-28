"""MCP Server base class for Geronimo models."""

import sys
import json
from typing import Optional
from abc import ABC, abstractmethod

from geronimo.mcp.tools import Tool, ToolResult


class MCPServer(ABC):
    """Base class for MCP servers exposing Geronimo models.
    
    Subclass this to create an MCP server for your model:
    
        class IrisMCPServer(MCPServer):
            name = "iris-classifier"
            version = "1.0.0"
            
            def get_tools(self) -> list[Tool]:
                return [
                    Tool(
                        name="predict",
                        description="Predict iris species",
                        input_schema={...},
                        handler=self.predict,
                    )
                ]
            
            def predict(self, features: dict) -> ToolResult:
                result = self.model.predict(features)
                return ToolResult.json(result)
    """
    
    name: str = "geronimo-model"
    version: str = "1.0.0"
    description: str = "Geronimo ML Model MCP Server"
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Register tools from get_tools()."""
        for tool in self.get_tools():
            self._tools[tool.name] = tool
    
    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """Return list of tools this server exposes.
        
        Override this to define your model's tools.
        """
        pass
    
    def handle_request(self, request: dict) -> dict:
        """Handle an incoming MCP request."""
        method = request.get("method", "")
        request_id = request.get("id")
        params = request.get("params", {})
        
        if method == "initialize":
            return self._handle_initialize(request_id, params)
        elif method == "tools/list":
            return self._handle_tools_list(request_id)
        elif method == "tools/call":
            return self._handle_tools_call(request_id, params)
        else:
            return self._error_response(request_id, -32601, f"Unknown method: {method}")
    
    def _handle_initialize(self, request_id: int, params: dict) -> dict:
        """Handle initialize request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "serverInfo": {
                    "name": self.name,
                    "version": self.version,
                },
            },
        }
    
    def _handle_tools_list(self, request_id: int) -> dict:
        """Handle tools/list request."""
        tools = [tool.to_mcp_schema() for tool in self._tools.values()]
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": tools},
        }
    
    def _handle_tools_call(self, request_id: int, params: dict) -> dict:
        """Handle tools/call request."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        if tool_name not in self._tools:
            return self._error_response(request_id, -32602, f"Unknown tool: {tool_name}")
        
        tool = self._tools[tool_name]
        result = tool.invoke(arguments)
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": result.content,
                "isError": result.is_error,
            },
        }
    
    def _error_response(self, request_id: int, code: int, message: str) -> dict:
        """Create an error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
    
    def run_stdio(self):
        """Run the server using stdio transport.
        
        This is the standard transport for local MCP servers.
        """
        print(f"Starting MCP server: {self.name} v{self.version}", file=sys.stderr)
        
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError as e:
                error = self._error_response(None, -32700, f"Parse error: {e}")
                print(json.dumps(error), flush=True)
            except Exception as e:
                error = self._error_response(None, -32603, f"Internal error: {e}")
                print(json.dumps(error), flush=True)
