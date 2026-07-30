"""Tests for geronimo.models.objective module."""

import pytest

from geronimo.models import Model, ObjectiveSpec, HyperParams


class TestObjectiveSpec:
    """Tests for ObjectiveSpec class."""

    def test_valid_named_objective_mse(self):
        """Test that 'mse' resolves to the correct sklearn key."""
        spec = ObjectiveSpec("mse")
        assert spec.name == "mse"
        assert spec.type == "named"
        assert spec.resolved == "squared_error"

    def test_valid_named_objective_mae(self):
        """Test that 'mae' resolves to the correct sklearn key."""
        spec = ObjectiveSpec("mae")
        assert spec.name == "mae"
        assert spec.resolved == "absolute_error"

    def test_valid_named_objective_binary_crossentropy(self):
        """Test that 'binary_crossentropy' resolves correctly."""
        spec = ObjectiveSpec("binary_crossentropy")
        assert spec.name == "binary_crossentropy"
        assert spec.resolved == "log_loss"

    def test_valid_named_objective_categorical_crossentropy(self):
        """Test that 'categorical_crossentropy' resolves correctly."""
        spec = ObjectiveSpec("categorical_crossentropy")
        assert spec.name == "categorical_crossentropy"
        assert spec.resolved == "log_loss"

    def test_invalid_named_objective_raises_value_error(self):
        """Test that an unsupported name raises ValueError with clear message."""
        with pytest.raises(ValueError, match="Unsupported objective name: 'foo'"):
            ObjectiveSpec("foo")

    def test_invalid_named_objective_lists_supported(self):
        """Test that the error message lists supported objectives."""
        with pytest.raises(ValueError, match="Supported objectives"):
            ObjectiveSpec("unsupported_loss")

    def test_custom_callable_accepted(self):
        """Test that a valid custom callable is accepted."""
        def custom_loss(y_true, y_pred):
            return ((y_true - y_pred) ** 2).mean()

        spec = ObjectiveSpec(custom_loss)
        assert spec.name == "custom_loss"
        assert spec.type == "custom"
        assert spec.resolved is custom_loss

    def test_custom_callable_with_description(self):
        """Test that custom callables accept an optional description."""
        def huber_loss(y_true, y_pred):
            diff = (y_true - y_pred).abs()
            return (diff ** 2).mean()

        spec = ObjectiveSpec(huber_loss, description="Huber loss for robust regression")
        assert spec.description == "Huber loss for robust regression"

    def test_custom_callable_too_few_params_raises(self):
        """Test that a callable with only one parameter raises ValueError."""
        def bad_fn(y_true):
            return 0.0

        with pytest.raises(ValueError, match="must accept at least two arguments"):
            ObjectiveSpec(bad_fn)

    def test_custom_callable_no_params_raises(self):
        """Test that a callable with no parameters raises ValueError."""
        def bad_fn():
            return 0.0

        with pytest.raises(ValueError, match="must accept at least two arguments"):
            ObjectiveSpec(bad_fn)

    def test_custom_callable_keyword_only_params_raises(self):
        """Test that a callable with keyword-only first param raises."""
        def bad_fn(*, y_true, y_pred):
            return 0.0

        with pytest.raises(ValueError, match="must accept.*as positional"):
            ObjectiveSpec(bad_fn)

    def test_custom_callable_keyword_only_second_param_raises(self):
        """Test that a callable with keyword-only second param raises."""
        def bad_fn(y_true, *, y_pred):
            return 0.0

        with pytest.raises(ValueError, match="must accept.*as positional"):
            ObjectiveSpec(bad_fn)

    def test_repr_named(self):
        """Test __repr__ for named objectives."""
        spec = ObjectiveSpec("mse")
        assert "mse" in repr(spec)
        assert "squared_error" in repr(spec)

    def test_repr_custom(self):
        """Test __repr__ for custom objectives."""
        def my_loss(y_true, y_pred):
            return 0.0

        spec = ObjectiveSpec(my_loss)
        assert "my_loss" in repr(spec)
        assert "custom" in repr(spec)

    def test_custom_callable_with_extra_params_accepted(self):
        """Test that a callable with extra optional params is accepted."""
        def weighted_loss(y_true, y_pred, sample_weight=None):
            if sample_weight is not None:
                return ((sample_weight * (y_true - y_pred) ** 2).sum() / sample_weight.sum())
            return ((y_true - y_pred) ** 2).mean()

        spec = ObjectiveSpec(weighted_loss)
        assert spec.name == "weighted_loss"
        assert spec.type == "custom"


class TestModelObjective:
    """Tests for Model's optional objective attribute."""

    def test_model_without_objective_backward_compatible(self):
        """Test that a Model without objective declared works as before."""
        class SimpleModel(Model):
            name = "simple"
            version = "1.0.0"

            def train(self, X, y, params):
                self._mean = y.mean()

            def predict(self, X):
                return [self._mean] * len(X)

        model = SimpleModel()
        assert model.objective is None
        assert model.name == "simple"

    def test_model_with_named_objective(self):
        """Test that a Model can declare a named objective."""
        class NamedModel(Model):
            name = "named"
            version = "1.0.0"
            objective = ObjectiveSpec("mse")

            def train(self, X, y, params):
                pass

            def predict(self, X):
                return []

        model = NamedModel()
        assert model.objective.name == "mse"
        assert model.objective.resolved == "squared_error"

    def test_model_with_custom_objective(self):
        """Test that a Model can declare a custom objective."""
        def custom(y_true, y_pred):
            return 0.0

        class CustomModel(Model):
            name = "custom"
            version = "1.0.0"
            objective = ObjectiveSpec(custom)

            def train(self, X, y, params):
                pass

            def predict(self, X):
                return []

        model = CustomModel()
        assert model.objective.type == "custom"

    def test_model_with_invalid_objective_raises_at_definition(self):
        """Test that an invalid objective raises ValueError at class definition."""
        with pytest.raises(ValueError, match="Unsupported objective name"):
            class BadModel(Model):
                name = "bad"
                version = "1.0.0"
                objective = ObjectiveSpec("nonexistent")

                def train(self, X, y, params):
                    pass

                def predict(self, X):
                    return []

    def test_model_subclass_can_override_objective(self):
        """Test that a subclass can override a parent's objective."""
        class ParentModel(Model):
            name = "parent"
            version = "1.0.0"
            objective = ObjectiveSpec("mse")

            def train(self, X, y, params):
                pass

            def predict(self, X):
                return []

        class ChildModel(ParentModel):
            name = "child"
            version = "2.0.0"
            objective = ObjectiveSpec("mae")

        parent = ParentModel()
        child = ChildModel()
        assert parent.objective.name == "mse"
        assert child.objective.name == "mae"


class TestObjectiveSpecIntegration:
    """Integration tests for ObjectiveSpec with Model."""

    def test_model_objective_attribute_default_none(self):
        """Test that the default objective is None for new subclasses."""
        class UnspecifiedModel(Model):
            name = "unspecified"
            version = "1.0.0"

            def train(self, X, y, params):
                pass

            def predict(self, X):
                return []

        model = UnspecifiedModel()
        assert model.objective is None

    def test_objective_spec_is_independent(self):
        """Test that ObjectiveSpec instances are independent."""
        spec1 = ObjectiveSpec("mse")
        spec2 = ObjectiveSpec("mae")
        assert spec1.resolved != spec2.resolved
        assert spec1.name != spec2.name
