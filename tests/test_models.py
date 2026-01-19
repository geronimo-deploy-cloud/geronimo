"""Tests for geronimo.models module."""

import pytest

from geronimo.models import Model, HyperParams


class TestHyperParams:
    """Tests for HyperParams class."""

    def test_basic_params(self):
        """Test basic hyperparameter creation."""
        params = HyperParams(learning_rate=0.01, n_estimators=100)
        assert params.learning_rate == 0.01
        assert params.n_estimators == 100

    def test_list_params(self):
        """Test list parameters for grid search."""
        params = HyperParams(n_estimators=[100, 200])
        # First value should be returned
        assert params.n_estimators == 100

    def test_params_to_dict(self):
        """Test converting params to dictionary."""
        params = HyperParams(learning_rate=0.01, batch_size=32)
        d = params.to_dict()
        assert d["learning_rate"] == 0.01
        assert d["batch_size"] == 32

    def test_params_grid(self):
        """Test grid search iteration."""
        params = HyperParams(a=[1, 2], b=[3, 4])
        combos = list(params.grid())
        assert len(combos) == 4  # 2 x 2 combinations


class TestModel:
    """Tests for Model base class."""

    def test_model_subclass(self):
        """Test creating a model subclass."""
        class TestModel(Model):
            name = "test-model"
            version = "1.0.0"

            def train(self, X, y, params):
                self.estimator = "trained"

            def predict(self, X):
                return [1] * len(X)

        model = TestModel()
        assert model.name == "test-model"
        assert model.version == "1.0.0"

    def test_model_train_predict(self, iris_df):
        """Test model train and predict."""
        class SimpleModel(Model):
            name = "simple"
            version = "1.0.0"

            def train(self, X, y, params):
                self._mean = y.mean()

            def predict(self, X):
                return [self._mean] * len(X)

        model = SimpleModel()
        X = iris_df.drop("target", axis=1)
        y = iris_df["target"]

        model.train(X, y, HyperParams())
        preds = model.predict(X)
        
        assert len(preds) == len(X)

    def test_model_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # Model without train/predict should fail
        with pytest.raises(TypeError):
            class IncompleteModel(Model):
                name = "incomplete"
                version = "1.0.0"
            
            IncompleteModel()
