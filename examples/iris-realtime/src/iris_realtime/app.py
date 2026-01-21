"""FastAPI application for Iris species prediction.

Run with:
    uvicorn iris_realtime.app:app --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from iris_realtime.sdk import IrisEndpoint, get_endpoint


# =============================================================================
# Request/Response Models
# =============================================================================

class IrisPredictRequest(BaseModel):
    """Request model for Iris prediction."""
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=10, description="Petal width in cm")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        }
    }


class IrisPredictResponse(BaseModel):
    """Response model for Iris prediction."""
    prediction: str = Field(..., description="Predicted species name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    probabilities: dict[str, float] = Field(..., description="Class probabilities")


# =============================================================================
# Application Setup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle - load model on startup."""
    # Startup: initialize endpoint (loads or trains model)
    endpoint = get_endpoint()
    print(f"Iris model loaded: {endpoint.model.name} v{endpoint.model.version}")
    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title="Iris Species Classifier",
    description="Predict Iris flower species from sepal/petal measurements",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Routes
# =============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/info")
def model_info():
    """Get model information."""
    endpoint = get_endpoint()
    return {
        "model": endpoint.model.name,
        "version": endpoint.model.version,
        "species": endpoint.model.SPECIES,
        "features": endpoint.model.features.feature_names,
    }


@app.post("/predict", response_model=IrisPredictResponse)
def predict(request: IrisPredictRequest):
    """Predict Iris species from flower measurements.
    
    Pass sepal and petal dimensions to classify as:
    - setosa
    - versicolor
    - virginica
    """
    try:
        endpoint = get_endpoint()
        result = endpoint.handle(request.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Example usage
# =============================================================================
# 
# curl http://localhost:8000/health
# curl http://localhost:8000/info
# curl -X POST http://localhost:8000/predict \
#      -H "Content-Type: application/json" \
#      -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
