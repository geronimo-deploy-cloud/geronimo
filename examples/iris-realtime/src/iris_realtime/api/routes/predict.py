"""Prediction endpoints using Geronimo SDK Endpoint."""

import time
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from iris_realtime.sdk.endpoint import PredictEndpoint

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize SDK endpoint
_endpoint = None


def get_endpoint() -> PredictEndpoint:
    """Get or initialize the SDK endpoint."""
    global _endpoint
    if _endpoint is None:
        _endpoint = PredictEndpoint()
        _endpoint.initialize()
        logger.info("SDK Endpoint initialized")
    return _endpoint


@router.post("/predict")
async def predict(request: dict[str, Any]) -> dict[str, Any]:
    """Generate predictions using the SDK Endpoint.

    Args:
        request: Input features for prediction.

    Returns:
        Model predictions with metadata.
    """
    start_time = time.perf_counter()

    try:
        endpoint = get_endpoint()
        result = endpoint.handle(request)

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Prediction completed in {latency_ms:.2f}ms")

        return {
            **result,
            "latency_ms": latency_ms,
        }

    except NotImplementedError as e:
        raise HTTPException(
            status_code=501, 
            detail=f"Endpoint not implemented: {e}"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
