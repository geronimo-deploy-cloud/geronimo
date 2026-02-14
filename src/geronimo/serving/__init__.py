"""Geronimo Serving Module.

The serving module provides the infrastructure for exposing trained models as
scalable API endpoints. It handles request parsing, feature transformation,
and model inference.

Key features:
- Fast API integration for high-performance serving.
- Automatic generation of OpenAPI (Swagger) documentation.
- Support for batch and real-time inference requests.
- Health checks and monitoring endpoints.

Endpoints are automatically Dockerized and deployed to the target infrastructure.
"""

from geronimo.serving.endpoint import Endpoint

__all__ = ["Endpoint"]

__docformat__ = "google"
