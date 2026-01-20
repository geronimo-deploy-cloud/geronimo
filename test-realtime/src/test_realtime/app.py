"""FastAPI application - thin wrapper around SDK endpoint.

This app integrates:
- SDK endpoint for predictions
- Monitoring middleware for latency/error tracking
- Metrics collector for CloudWatch/custom backends
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from test_realtime.sdk.endpoint import PredictEndpoint
from test_realtime.monitoring.middleware import MonitoringMiddleware
from test_realtime.monitoring.metrics import MetricsCollector


# =============================================================================
# Configuration - customize these values
# =============================================================================

PROJECT_NAME = "test-realtime"

# Metrics backend: "cloudwatch", "local", or custom
METRICS_BACKEND = "local"  # TODO: Change to "cloudwatch" for production


# =============================================================================
# Initialize components
# =============================================================================

# Initialize metrics collector
# For CloudWatch: MetricsCollector(project_name=PROJECT_NAME, namespace="MLModels")
metrics = MetricsCollector(project_name=PROJECT_NAME)

# Lazy-load endpoint
_endpoint = None


def get_endpoint():
    global _endpoint
    if _endpoint is None:
        _endpoint = PredictEndpoint()
        _endpoint.load()
    return _endpoint


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle - load model on startup."""
    # Startup: pre-load model for faster first request
    get_endpoint()
    yield
    # Shutdown: cleanup if needed


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title=PROJECT_NAME,
    description="ML model serving API with monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - customize origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monitoring middleware - tracks latency, errors, request counts
app.add_middleware(MonitoringMiddleware, collector=metrics)


# =============================================================================
# Request/Response Models
# =============================================================================

class PredictRequest(BaseModel):
    """Prediction request schema."""
    features: dict[str, Any]


class PredictResponse(BaseModel):
    """Prediction response schema."""
    prediction: Any


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics():
    """Get current metrics summary.
    
    Returns latency percentiles, request counts, and error rates.
    """
    return {
        "latency_p50_ms": metrics.get_latency_p50(),
        "latency_p99_ms": metrics.get_latency_p99(),
        "request_count": metrics.get_request_count(),
        "error_count": metrics.get_error_count(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Generate prediction from model.
    
    The endpoint handles:
    1. preprocess() - transform request to features
    2. model.predict() - generate prediction
    3. postprocess() - format response
    
    Latency and errors are automatically tracked by MonitoringMiddleware.
    """
    try:
        endpoint = get_endpoint()
        result = endpoint.handle(request.model_dump())
        return PredictResponse(prediction=result)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
