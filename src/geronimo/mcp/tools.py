"""MCP Tool definitions."""

from typing import Any, Callable
from dataclasses import dataclass, field
from pydantic import BaseModel


class ToolResult(BaseModel):
    """Result from an MCP tool invocation."""
    
    content: list[dict]  # MCP content blocks
    is_error: bool = False
    
    @classmethod
    def text(cls, text: str) -> "ToolResult":
        """Create a text result."""
        return cls(content=[{"type": "text", "text": text}])
    
    @classmethod
    def json(cls, data: dict) -> "ToolResult":
        """Create a JSON result."""
        import json
        return cls(content=[{"type": "text", "text": json.dumps(data, indent=2)}])
    
    @classmethod
    def error(cls, message: str) -> "ToolResult":
        """Create an error result."""
        return cls(content=[{"type": "text", "text": f"Error: {message}"}], is_error=True)


@dataclass
class Tool:
    """MCP Tool definition.
    
    Example:
        @Tool.define(
            name="predict",
            description="Run model prediction",
            input_schema={
                "type": "object",
                "properties": {
                    "features": {"type": "object"}
                }
            }
        )
        def predict(features: dict) -> ToolResult:
            ...
    """
    
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., ToolResult]
    
    @classmethod
    def define(cls, name: str, description: str, input_schema: dict):
        """Decorator to define a tool."""
        def decorator(func: Callable[..., ToolResult]) -> "Tool":
            return cls(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=func,
            )
        return decorator
    
    def to_mcp_schema(self) -> dict:
        """Convert to MCP tool schema format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
    
    def invoke(self, arguments: dict) -> ToolResult:
        """Invoke the tool with given arguments."""
        try:
            return self.handler(**arguments)
        except Exception as e:
            return ToolResult.error(str(e))
