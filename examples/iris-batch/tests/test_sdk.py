"""Tests for SDK components."""

import pytest


class TestProjectModel:
    """Tests for ProjectModel."""

    def test_model_import(self):
        """Test model can be imported."""
        from iris_batch.sdk.model import IrisBatchModel
        
        model = IrisBatchModel()
        assert model.name == "iris-batch"


class TestProjectFeatures:
    """Tests for ProjectFeatures."""

    def test_features_import(self):
        """Test features can be imported."""
        from iris_batch.sdk.features import IrisBatchFeatures
        
        features = IrisBatchFeatures()
        assert features is not None
