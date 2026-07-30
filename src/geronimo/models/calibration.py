"""Model calibration support for probability calibration.

Provides opt-in probability calibration for classification models via
scikit-learn's CalibratedClassifierCV. Supports Platt scaling (sigmoid)
and isotonic regression.

Design Rationale:
- Calibration is declared as an optional attribute on Model, consistent
  with how `features` and `estimator` are declared.
- CalibrationSpec is a simple class (matching HyperParams style, not
  Pydantic) that holds method, cv, and alpha parameters.
- Since no SplitSpec exists in the codebase, calibration data is a
  simple internal 80/20 train/calibration split within train().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CalibrationSpec:
    """Specification for probability calibration.

    Wraps scikit-learn's CalibratedClassifierCV. Calibration is
    **opt-in** — when None, models train and predict exactly as
    they do today.

    Args:
        method: Calibration method — ``"sigmoid"`` (Platt scaling)
            or ``"isotonic"`` (isotonic regression).
        cv: Cross-validation fold specification. Passed directly to
            CalibratedClassifierCV. ``None`` means the wrapped
            estimator's ``predict_proba`` is used directly on a
            held-out split (no CV averaging).

    Note:
        The ``alpha`` parameter from the deprecated ``CalibratedClassifier``
        is not supported by ``CalibratedClassifierCV`` and has been removed.
    """

    method: str = "sigmoid"
    """Calibration method: ``"sigmoid"`` or ``"isotonic"``."""

    cv: Optional[int] = None
    """Cross-validation fold count. ``None`` = no CV averaging."""

    _VALID_METHODS = frozenset({"sigmoid", "isotonic"})

    def __post_init__(self) -> None:
        """Validate calibration specification."""
        if self.method not in self._VALID_METHODS:
            raise ValueError(
                f"Invalid calibration method '{self.method}'. "
                f"Must be one of: {sorted(self._VALID_METHODS)}"
            )
        if self.cv is not None and (not isinstance(self.cv, int) or self.cv < 1):
            raise ValueError(
                f"cv must be a positive int or None, got {self.cv}"
            )

    def __repr__(self) -> str:
        cv_str = str(self.cv) if self.cv is not None else "None"
        return f"CalibrationSpec(method='{self.method}', cv={cv_str})"


def _build_calibrated_estimator(
    estimator: Any,
    calibration: CalibrationSpec,
) -> Any:
    """Wrap an estimator with CalibratedClassifierCV.

    Args:
        estimator: A fitted sklearn-compatible classifier with
            ``predict_proba``.
        calibration: Calibration specification.

    Returns:
        A CalibratedClassifierCV instance ready to be fit on
        held-out calibration data.
    """
    from sklearn.calibration import CalibratedClassifierCV

    return CalibratedClassifierCV(
        estimator=estimator,
        method=calibration.method,
        cv=calibration.cv,
        n_jobs=1,
    )


def _clamp_probabilities(probs: Any) -> Any:
    """Clamp predicted probabilities to [0, 1].

    CalibratedClassifierCV should always produce valid probabilities,
    but this acts as a safety net.

    Args:
        probs: Probability array from predict_proba.

    Returns:
        Probabilities clamped to [0, 1].
    """
    import numpy as np

    clamped = np.clip(probs, 0.0, 1.0)
    # Ensure rows sum to 1 (numerical fix for clamping artifacts)
    row_sums = clamped.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return clamped / row_sums
