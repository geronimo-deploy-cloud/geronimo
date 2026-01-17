"""FastAPI application for verification-project ML serving."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from verification_project.api.routes import health, predict
from verification_project.ml.predictor import ModelPredictor
from verification_project.monitoring.middleware import MonitoringMiddleware
from verification_project.monitoring.metrics import MetricsCollector
from verification_project.api import deps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize metrics collector
metrics = MetricsCollector(project_name="verification-project")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for model loading."""
    logger.info("Loading model...")
    deps.predictor = ModelPredictor()
    deps.predictor.load()
    logger.info("Model loaded successfully")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="verification-project",
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

# Monitoring middleware
app.add_middleware(MonitoringMiddleware, collector=metrics)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/v1", tags=["Predictions"])
