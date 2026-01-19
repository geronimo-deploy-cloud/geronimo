"""FastAPI application for iris-realtime ML serving."""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from iris_realtime.api.routes import health, predict
from iris_realtime.ml.predictor import ModelPredictor
from iris_realtime.monitoring.middleware import MonitoringMiddleware
from iris_realtime.monitoring.metrics import MetricsCollector
from iris_realtime.api import deps
from iris_realtime.agent.server import mcp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize metrics collector
metrics = MetricsCollector(project_name="iris-realtime")


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
    title="iris-realtime",
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

# Mount MCP Agent (Streamable HTTP)
if os.getenv("ENABLE_MCP_AGENT", "true").lower() == "true":
    app.mount("/mcp", mcp.streamable_http_app())

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/v1", tags=["Predictions"])
