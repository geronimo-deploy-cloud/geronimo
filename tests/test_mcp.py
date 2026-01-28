"""Tests for geronimo.mcp module."""

import pytest
import json
from unittest.mock import MagicMock, patch

from geronimo.mcp.tools import Tool, ToolResult
from geronimo.mcp.server import MCPServer


# =============================================================================
# ToolResult Tests
# =============================================================================

class TestToolResult:
    """Tests for ToolResult class."""
    
    def test_text_result(self):
        """Test creating a text result."""
        result = ToolResult.text("Hello, world!")
        
        assert len(result.content) == 1
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "Hello, world!"
        assert result.is_error is False
    
    def test_json_result(self):
        """Test creating a JSON result."""
        data = {"species": "setosa", "confidence": 0.95}
        result = ToolResult.json(data)
        
        assert len(result.content) == 1
        assert result.content[0]["type"] == "text"
        # Verify it's valid JSON
        parsed = json.loads(result.content[0]["text"])
        assert parsed["species"] == "setosa"
        assert parsed["confidence"] == 0.95
        assert result.is_error is False
    
    def test_error_result(self):
        """Test creating an error result."""
        result = ToolResult.error("Something went wrong")
        
        assert len(result.content) == 1
        assert "Error: Something went wrong" in result.content[0]["text"]
        assert result.is_error is True
    
    def test_custom_content(self):
        """Test creating result with custom content."""
        content = [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]
        result = ToolResult(content=content)
        
        assert len(result.content) == 2
        assert result.is_error is False


# =============================================================================
# Tool Tests
# =============================================================================

class TestTool:
    """Tests for Tool class."""
    
    def test_tool_creation(self):
        """Test creating a tool."""
        def handler(x: int) -> ToolResult:
            return ToolResult.json({"result": x * 2})
        
        tool = Tool(
            name="double",
            description="Double a number",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            handler=handler,
        )
        
        assert tool.name == "double"
        assert tool.description == "Double a number"
        assert tool.input_schema["type"] == "object"
    
    def test_tool_invoke(self):
        """Test invoking a tool."""
        def handler(value: str) -> ToolResult:
            return ToolResult.text(f"Received: {value}")
        
        tool = Tool(
            name="echo",
            description="Echo back",
            input_schema={},
            handler=handler,
        )
        
        result = tool.invoke({"value": "hello"})
        
        assert result.is_error is False
        assert "Received: hello" in result.content[0]["text"]
    
    def test_tool_invoke_handles_exception(self):
        """Test that tool invoke catches exceptions."""
        def handler() -> ToolResult:
            raise ValueError("Test error")
        
        tool = Tool(
            name="failing",
            description="Always fails",
            input_schema={},
            handler=handler,
        )
        
        result = tool.invoke({})
        
        assert result.is_error is True
        assert "Test error" in result.content[0]["text"]
    
    def test_tool_to_mcp_schema(self):
        """Test converting tool to MCP schema format."""
        tool = Tool(
            name="predict",
            description="Make prediction",
            input_schema={
                "type": "object",
                "properties": {"features": {"type": "array"}},
            },
            handler=lambda: ToolResult.text("ok"),
        )
        
        schema = tool.to_mcp_schema()
        
        assert schema["name"] == "predict"
        assert schema["description"] == "Make prediction"
        assert schema["inputSchema"]["type"] == "object"
    
    def test_tool_define_decorator(self):
        """Test Tool.define decorator."""
        @Tool.define(
            name="greet",
            description="Greet someone",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        )
        def greet(name: str) -> ToolResult:
            return ToolResult.text(f"Hello, {name}!")
        
        assert isinstance(greet, Tool)
        assert greet.name == "greet"
        
        result = greet.invoke({"name": "Alice"})
        assert "Hello, Alice!" in result.content[0]["text"]


# =============================================================================
# MCPServer Tests
# =============================================================================

class SimpleMCPServer(MCPServer):
    """Simple MCP server for testing."""
    
    name = "test-server"
    version = "1.0.0"
    description = "Test MCP Server"
    
    def get_tools(self) -> list[Tool]:
        return [
            Tool(
                name="echo",
                description="Echo back input",
                input_schema={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
                handler=self.echo,
            ),
            Tool(
                name="add",
                description="Add two numbers",
                input_schema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
                handler=self.add,
            ),
        ]
    
    def echo(self, message: str) -> ToolResult:
        return ToolResult.text(f"Echo: {message}")
    
    def add(self, a: float, b: float) -> ToolResult:
        return ToolResult.json({"sum": a + b})


class TestMCPServer:
    """Tests for MCPServer class."""
    
    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return SimpleMCPServer()
    
    def test_server_initialization(self, server):
        """Test server initializes with tools."""
        assert server.name == "test-server"
        assert server.version == "1.0.0"
        assert len(server._tools) == 2
        assert "echo" in server._tools
        assert "add" in server._tools
    
    def test_handle_initialize(self, server):
        """Test handling initialize request."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }
        
        response = server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["serverInfo"]["name"] == "test-server"
        assert response["result"]["serverInfo"]["version"] == "1.0.0"
        assert "capabilities" in response["result"]
    
    def test_handle_tools_list(self, server):
        """Test handling tools/list request."""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        
        response = server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        assert "tools" in response["result"]
        
        tools = response["result"]["tools"]
        assert len(tools) == 2
        
        tool_names = [t["name"] for t in tools]
        assert "echo" in tool_names
        assert "add" in tool_names
    
    def test_handle_tools_call_echo(self, server):
        """Test calling echo tool."""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "echo",
                "arguments": {"message": "Hello MCP!"},
            },
        }
        
        response = server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert "result" in response
        assert response["result"]["isError"] is False
        assert "Echo: Hello MCP!" in response["result"]["content"][0]["text"]
    
    def test_handle_tools_call_add(self, server):
        """Test calling add tool."""
        request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "add",
                "arguments": {"a": 10, "b": 5},
            },
        }
        
        response = server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 4
        assert "result" in response
        assert response["result"]["isError"] is False
        
        # Parse JSON content
        content = json.loads(response["result"]["content"][0]["text"])
        assert content["sum"] == 15
    
    def test_handle_unknown_tool(self, server):
        """Test calling unknown tool returns error."""
        request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "nonexistent",
                "arguments": {},
            },
        }
        
        response = server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 5
        assert "error" in response
        assert "Unknown tool" in response["error"]["message"]
    
    def test_handle_unknown_method(self, server):
        """Test unknown method returns error."""
        request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "unknown/method",
            "params": {},
        }
        
        response = server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 6
        assert "error" in response
        assert response["error"]["code"] == -32601
    
    def test_error_response_format(self, server):
        """Test error response has correct format."""
        response = server._error_response(99, -32600, "Invalid request")
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 99
        assert response["error"]["code"] == -32600
        assert response["error"]["message"] == "Invalid request"


# =============================================================================
# Module Import Tests
# =============================================================================

class TestModuleExports:
    """Tests for module public API."""
    
    def test_mcp_module_exports(self):
        """Test that mcp module exports expected symbols."""
        from geronimo import mcp
        
        assert hasattr(mcp, "MCPServer")
        assert hasattr(mcp, "Tool")
        assert hasattr(mcp, "ToolResult")
    
    def test_direct_imports(self):
        """Test direct imports work."""
        from geronimo.mcp import MCPServer, Tool, ToolResult
        
        assert MCPServer is not None
        assert Tool is not None
        assert ToolResult is not None
