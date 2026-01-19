"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for predictions."""

    features: dict[str, float | int | str | list] = Field(
        ...,
        description="Input features as key-value pairs",
        examples=[{"feature_1": 1.5, "feature_2": "category_a", "feature_3": [1, 2, 3]}],
    )


class PredictionResponse(BaseModel):
    """Response schema for predictions."""

    prediction: float | int | str | list = Field(
        ...,
        description="Model prediction result",
    )
    model_version: str = Field(
        ...,
        description="Version of the model used for prediction",
    )
    latency_ms: float = Field(
        ...,
        description="Prediction latency in milliseconds",
    )


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str = Field(..., description="Error message")
