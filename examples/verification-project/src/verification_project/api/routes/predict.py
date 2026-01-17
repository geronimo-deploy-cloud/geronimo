"""Prediction endpoints."""

import time
import logging

from fastapi import APIRouter, HTTPException, Depends

from verification_project.api.models.schemas import (
    PredictionRequest,
    PredictionResponse,
)
from verification_project.api.deps import get_predictor
from verification_project.ml.predictor import ModelPredictor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    predictor: ModelPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Generate predictions for the input features.

    Args:
        request: Input features for prediction.
        predictor: The loaded model predictor (injected).

    Returns:
        Model predictions with metadata.
    """
    start_time = time.perf_counter()

    try:
        prediction = predictor.predict(request.features)

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Prediction completed in {latency_ms:.2f}ms")

        return PredictionResponse(
            prediction=prediction,
            model_version=predictor.version,
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
