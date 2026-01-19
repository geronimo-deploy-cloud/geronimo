"""MCP Server implementation for the ML model."""

from typing import Any
from mcp.server.fastmcp import FastMCP

# We need to import the project specific modules. 
# These imports will be fixed by the generator to match the project name.
from iris_realtime.api.models.schemas import PredictionRequest
from iris_realtime.api.deps import get_predictor

# Initialize FastMCP Server
# "name" will be replaced by the generator
mcp = FastMCP("geronimo-agent")

@mcp.tool()
async def predict(input_data: dict[str, Any]) -> str:
    """Make a prediction using the ML model.
    
    Args:
        input_data: The input features for the model.
    """
    predictor = get_predictor()
    
    # Validate input using Pydantic model
    try:
        request = PredictionRequest(**input_data)
    except Exception as e:
        return f"Error validating input: {str(e)}"

    try:
        prediction = predictor.predict(request)
        return str(prediction.dict())
    except Exception as e:
        return f"Prediction error: {str(e)}"

def run_stdio():
    """Run the server using stdio transport."""
    mcp.run(transport="stdio")

if __name__ == "__main__":
    run_stdio()
