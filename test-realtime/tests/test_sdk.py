"""Tests for SDK components."""

import pytest


class TestProjectModel:
    """Tests for ProjectModel."""

    def test_model_import(self):
        """Test model can be imported."""
        from test_realtime.sdk.model import ProjectModel
        
        model = ProjectModel()
        assert model.name == "test-realtime"


class TestProjectFeatures:
    """Tests for ProjectFeatures."""

    def test_features_import(self):
        """Test features can be imported."""
        from test_realtime.sdk.features import ProjectFeatures
        
        features = ProjectFeatures()
        assert features is not None
