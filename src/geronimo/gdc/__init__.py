"""Geronimo Deploy Cloud Integration.

This module provides the client and target definitions for interacting with the
managed Geronimo Deploy Cloud platform. It enables users to deploy their models
without managing their own infrastructure.

Key components:
- GeronimoCloudClient: A client for the Geronimo Deploy Cloud API.
- GeronimoCloudTarget: A deployment target that routes requests to the managed service.

Use this module when you want a fully managed deployment experience instead of
provisioning your own AWS resources.
"""

from geronimo.gdc.client import GeronimoCloudClient
from geronimo.gdc.target import GeronimoCloudTarget

__all__ = ["GeronimoCloudClient", "GeronimoCloudTarget"]

__docformat__ = "google"
