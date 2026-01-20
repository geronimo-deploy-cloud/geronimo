"""FastAPI application - thin wrapper around SDK endpoint.

Credit Risk API for loan default probability prediction.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from credit_risk.sdk.endpoint import PredictEndpoint
from credit_risk.monitoring.middleware import MonitoringMiddleware
from credit_risk.monitoring.metrics import MetricsCollector


# Configuration
PROJECT_NAME = "credit-risk"
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
    """Load model on startup."""
    get_endpoint()
    yield


# FastAPI App
app = FastAPI(
    title="Credit Risk API",
    description="Loan default probability prediction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(MonitoringMiddleware, collector=metrics)


# Request/Response Models
class LoanApplication(BaseModel):
    """Loan application features."""
    income: float
    debt_ratio: float
    credit_score: int
    loan_amount: float
    employment_type: str = "employed"
    loan_purpose: str = "personal"


class RiskAssessment(BaseModel):
    """Risk assessment response."""
    default_probability: float
    risk_level: str
    recommendation: str


# Endpoints
@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics():
    """Get current metrics."""
    return {
        "latency_p50_ms": metrics.get_latency_p50(),
        "latency_p99_ms": metrics.get_latency_p99(),
        "request_count": metrics.get_request_count(),
        "error_count": metrics.get_error_count(),
    }


@app.post("/assess", response_model=RiskAssessment)
def assess_risk(application: LoanApplication):
    """Assess credit risk for a loan application.
    
    Returns default probability and recommendation.
    """
    try:
        endpoint = get_endpoint()
        result = endpoint.handle({"features": application.model_dump()})
        return RiskAssessment(**result)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
