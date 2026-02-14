"""Tests for SDK components."""

import pytest


class TestProjectModel:
    """Tests for ProjectModel."""

    def test_model_import(self):
        """Test model can be imported."""
        from iris_realtime.sdk.model import IrisRealtimeModel
        
        model = IrisRealtimeModel()
        assert model.name == "iris-realtime"


class TestProjectFeatures:
    """Tests for ProjectFeatures."""

    def test_features_import(self):
        """Test features can be imported."""
        from iris_realtime.sdk.features import IrisRealtimeFeatures
        
        features = IrisRealtimeFeatures()
        assert features is not None
