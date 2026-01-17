"""FastAPI application for test-ml-project ML serving."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from test_ml_project.api.routes import health, predict
from test_ml_project.ml.predictor import ModelPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global model instance (loaded at startup)
predictor: ModelPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for model loading."""
    global predictor

    logger.info("Loading model...")
    predictor = ModelPredictor()
    predictor.load()
    logger.info("Model loaded successfully")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="test-ml-project",
    description="ML model serving API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/v1", tags=["Predictions"])


def get_predictor() -> ModelPredictor:
    """Get the loaded model predictor."""
    if predictor is None:
        raise RuntimeError("Model not loaded")
    return predictor
