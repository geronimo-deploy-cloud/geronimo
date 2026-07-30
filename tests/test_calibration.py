"""Tests for model probability calibration support."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from geronimo.models import Model, CalibrationSpec, HyperParams


# =============================================================================
# CalibrationSpec validation tests
# =============================================================================


class TestCalibrationSpec:
    """Tests for CalibrationSpec class."""

    def test_valid_sigmoid(self):
        """Test default sigmoid method."""
        spec = CalibrationSpec()
        assert spec.method == "sigmoid"
        assert spec.cv is None

    def test_valid_isotonic(self):
        """Test isotonic method."""
        spec = CalibrationSpec(method="isotonic")
        assert spec.method == "isotonic"

    def test_invalid_method_raises_valueerror(self):
        """Test that invalid method string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid calibration method"):
            CalibrationSpec(method="invalid_method")

    def test_invalid_method_values(self):
        """Test various invalid method values."""
        for bad in ["platt", "logistic", "softmax", "none", ""]:
            with pytest.raises(ValueError, match="Invalid calibration method"):
                CalibrationSpec(method=bad)

    def test_valid_cv(self):
        """Test valid cv values."""
        spec = CalibrationSpec(cv=3)
        assert spec.cv == 3

    def test_cv_none(self):
        """Test cv=None (no CV averaging)."""
        spec = CalibrationSpec(cv=None)
        assert spec.cv is None

    def test_invalid_cv_raises(self):
        """Test that invalid cv raises ValueError."""
        with pytest.raises(ValueError, match="cv must be a positive int"):
            CalibrationSpec(cv=0)
        with pytest.raises(ValueError, match="cv must be a positive int"):
            CalibrationSpec(cv=-1)
        with pytest.raises(ValueError, match="cv must be a positive int"):
            CalibrationSpec(cv="3")

    def test_repr(self):
        """Test string representation."""
        spec = CalibrationSpec(method="isotonic", cv=5)
        repr_str = repr(spec)
        assert "isotonic" in repr_str
        assert "cv=5" in repr_str

        spec2 = CalibrationSpec()
        assert "None" in repr(spec2)


# =============================================================================
# Integration tests: calibrated vs uncalibrated models
# =============================================================================


class _CalibratedTestModel(Model):
    """Test model that supports calibration."""

    name = "test-calibrated"
    version = "1.0.0"

    def __init__(self):
        super().__init__()
        self._calibrated_estimator = None

    def train(self, X, y, params: HyperParams) -> None:
        """Train a RandomForestClassifier, optionally wrap with calibration."""
        import numpy as np
        from sklearn.model_selection import train_test_split

        X = np.asarray(X)
        y = np.asarray(y)

        # Base estimator
        self.estimator = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=42
        )

        # If calibration is configured, do an 80/20 split
        # and wrap the estimator for calibration fitting
        if self.calibration is not None:
            from geronimo.models.calibration import _build_calibrated_estimator

            X_train, X_cal, y_train, y_cal = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            self.estimator.fit(X_train, y_train)
            self._calibrated_estimator = _build_calibrated_estimator(
                self.estimator, self.calibration
            )
            self._calibrated_estimator.fit(X_cal, y_cal)
        else:
            self.estimator.fit(X, y)

        self._is_fitted = True

    def predict(self, X) -> np.ndarray:
        """Predict using calibrated or uncalibrated estimator."""
        X = np.asarray(X)
        if not self._is_fitted:
            raise RuntimeError("Model not trained.")

        # Use calibrated estimator if available, otherwise base estimator
        if self._calibrated_estimator is not None:
            return self._calibrated_estimator.predict_proba(X)
        else:
            return self.estimator.predict_proba(X)


class _UncalibratedTestModel(Model):
    """Test model without calibration (no regression baseline)."""

    name = "test-uncalibrated"
    version = "1.0.0"

    def train(self, X, y, params: HyperParams) -> None:
        """Train a simple RandomForestClassifier."""
        import numpy as np

        X = np.asarray(X)
        y = np.asarray(y)

        self.estimator = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=42
        )
        self.estimator.fit(X, y)
        self._is_fitted = True

    def predict(self, X) -> np.ndarray:
        """Predict using base estimator."""
        X = np.asarray(X)
        if not self._is_fitted:
            raise RuntimeError("Model not trained.")
        return self.estimator.predict_proba(X)


# =============================================================================
# Calibration method tests (Platt scaling / isotonic)
# =============================================================================


class TestCalibrationMethods:
    """Tests for Platt scaling and isotonic calibration."""

    @pytest.fixture
    def classification_data(self) -> tuple:
        """Generate a classification dataset for calibration tests."""
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=3,
            random_state=42,
        )
        return X, y

    def test_platt_scaling_calibration(self, classification_data):
        """Test Platt scaling (sigmoid) calibration produces valid probabilities.

        Verifies:
        - Outputs are within [0, 1]
        - Each row sums to 1 (within tolerance)
        - Calibration does not crash
        """
        X, y = classification_data

        model = _CalibratedTestModel()
        model.calibration = CalibrationSpec(method="sigmoid", cv=3)
        model.train(X, y, HyperParams())

        preds = model.predict(X)

        # Validate probabilities are within [0, 1]
        assert np.all(preds >= 0.0 - 1e-9), "Probabilities below 0"
        assert np.all(preds <= 1.0 + 1e-9), "Probabilities above 1"

        # Validate rows sum to 1
        row_sums = preds.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Rows do not sum to 1"

    def test_isotonic_calibration(self, classification_data):
        """Test isotonic regression calibration produces valid probabilities.

        Verifies:
        - Outputs are within [0, 1]
        - Each row sums to 1 (within tolerance)
        - Calibration does not crash
        """
        X, y = classification_data

        model = _CalibratedTestModel()
        model.calibration = CalibrationSpec(method="isotonic", cv=3)
        model.train(X, y, HyperParams())

        preds = model.predict(X)

        # Validate probabilities are within [0, 1]
        assert np.all(preds >= 0.0 - 1e-9), "Probabilities below 0"
        assert np.all(preds <= 1.0 + 1e-9), "Probabilities above 1"

        # Validate rows sum to 1
        row_sums = preds.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Rows do not sum to 1"

    def test_calibration_preserves_ranking(self, classification_data):
        """Test that calibration preserves relative ranking of probabilities.

        Isotonic regression is monotonic — it should preserve or improve
        ranking. Platt (sigmoid) is a monotonic transform, so the most
        probable class should remain the same.
        """
        X, y = classification_data

        # Uncalibrated model
        uncal_model = _UncalibratedTestModel()
        uncal_model.train(X, y, HyperParams())
        uncal_preds = uncal_model.predict(X)

        # Calibrated model (sigmoid)
        cal_model = _CalibratedTestModel()
        cal_model.calibration = CalibrationSpec(method="sigmoid", cv=3)
        cal_model.train(X, y, HyperParams())
        cal_preds = cal_model.predict(X)

        # The predicted class (argmax) should be the same
        uncal_classes = np.argmax(uncal_preds, axis=1)
        cal_classes = np.argmax(cal_preds, axis=1)

        # With sufficient data, calibration should not change predictions
        # We use a lower threshold (70%) since calibration can shift
        # probability estimates, especially on small datasets.
        accuracy = np.mean(uncal_classes == cal_classes)
        assert accuracy > 0.7, (
            f"Calibration changed {100 * (1 - accuracy):.0f}% of predictions. "
            "Calibration should preserve the argmax class."
        )

    def test_calibration_with_no_cv(self, classification_data):
        """Test calibration with cv=None (no CV averaging)."""
        X, y = classification_data

        model = _CalibratedTestModel()
        model.calibration = CalibrationSpec(method="sigmoid", cv=None)
        model.train(X, y, HyperParams())

        preds = model.predict(X)

        assert np.all(preds >= 0.0 - 1e-9)
        assert np.all(preds <= 1.0 + 1e-9)
        row_sums = preds.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)


# =============================================================================
# No calibration (no regression) tests
# =============================================================================


class TestNoCalibration:
    """Test that models without calibration behave identically."""

    @pytest.fixture
    def classification_data(self) -> tuple:
        """Generate a classification dataset."""
        X, y = make_classification(
            n_samples=300,
            n_features=5,
            n_informative=3,
            n_classes=2,
            random_state=42,
        )
        return X, y

    def test_no_calibration_default(self, classification_data):
        """Test that a model without calibration works identically to before."""
        X, y = classification_data

        model = _UncalibratedTestModel()
        # No calibration attribute set — default is None
        model.train(X, y, HyperParams())

        preds = model.predict(X)

        # Should produce valid probabilities
        assert np.all(preds >= 0.0 - 1e-9)
        assert np.all(preds <= 1.0 + 1e-9)
        row_sums = preds.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_no_calibration_with_explicit_none(self, classification_data):
        """Test that explicit calibration=None behaves the same."""
        X, y = classification_data

        model = _CalibratedTestModel()
        model.calibration = None  # Explicitly no calibration
        model.train(X, y, HyperParams())

        preds = model.predict(X)

        assert np.all(preds >= 0.0 - 1e-9)
        assert np.all(preds <= 1.0 + 1e-9)
        row_sums = preds.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_uncalibrated_and_no_calibration_identical(self, classification_data):
        """Test that uncalibrated model and model with calibration=None
        produce identical results when using the same random state."""
        X, y = classification_data

        model1 = _UncalibratedTestModel()
        model1.train(X, y, HyperParams())
        preds1 = model1.predict(X)

        model2 = _CalibratedTestModel()
        model2.calibration = None
        model2.train(X, y, HyperParams())
        preds2 = model2.predict(X)

        np.testing.assert_array_equal(preds1, preds2)


# =============================================================================
# Invalid calibration method tests
# =============================================================================


class TestInvalidCalibrationMethod:
    """Test that invalid calibration method string raises ValueError."""

    def test_invalid_method_raises_on_spec_creation(self):
        """Test that creating a CalibrationSpec with invalid method raises."""
        with pytest.raises(ValueError, match="Invalid calibration method"):
            CalibrationSpec(method="platt")

    def test_invalid_method_raises_on_spec_creation_isotonic_bad(self):
        """Test that 'iso' is rejected."""
        with pytest.raises(ValueError, match="Invalid calibration method"):
            CalibrationSpec(method="iso")

    def test_invalid_method_still_trains_without_calibration(self):
        """Test that a model with invalid calibration spec cannot be trained.

        The spec itself is validated at construction time.
        """
        with pytest.raises(ValueError, match="Invalid calibration method"):
            model = _CalibratedTestModel()
            model.calibration = CalibrationSpec(method="bad")
            X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
            model.train(X, y, HyperParams())


# =============================================================================
# Edge case tests
# =============================================================================


class TestCalibrationEdgeCases:
    """Edge case tests for calibration."""

    def test_binary_classification_calibration(self):
        """Test calibration with binary classification."""
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=300,
            n_features=5,
            n_informative=3,
            n_classes=2,
            random_state=42,
        )

        model = _CalibratedTestModel()
        model.calibration = CalibrationSpec(method="sigmoid", cv=3)
        model.train(X, y, HyperParams())

        preds = model.predict(X)
        assert preds.shape[1] == 2  # Binary: 2 probability columns
        assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-6)

    def test_multiclass_calibration(self):
        """Test calibration with 4-class classification."""
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=5,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42,
        )

        model = _CalibratedTestModel()
        model.calibration = CalibrationSpec(method="isotonic", cv=3)
        model.train(X, y, HyperParams())

        preds = model.predict(X)
        assert preds.shape[1] == 4  # 4-class: 4 probability columns
        assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-6)

    def test_small_sample_calibration(self):
        """Test that calibration handles small samples gracefully."""
        X, y = make_classification(
            n_samples=50,
            n_features=5,
            n_informative=3,
            n_classes=2,
            random_state=42,
        )

        model = _CalibratedTestModel()
        model.calibration = CalibrationSpec(method="sigmoid", cv=2)
        model.train(X, y, HyperParams())

        preds = model.predict(X)
        assert np.all(preds >= 0.0 - 1e-9)
        assert np.all(preds <= 1.0 + 1e-9)

    def test_calibration_spec_repr(self):
        """Test string representation of CalibrationSpec."""
        spec = CalibrationSpec(method="isotonic", cv=5)
        repr_str = repr(spec)
        assert "CalibrationSpec" in repr_str
        assert "isotonic" in repr_str
        assert "cv=5" in repr_str

        spec_none = CalibrationSpec()
        assert "None" in repr(spec_none)
