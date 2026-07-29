"""Objective specification for ML model training.

Defines how a model's training objective (loss function) is declared,
validated, and resolved to the underlying framework's loss function.

Design rationale:
- ``objective`` is an optional class attribute on ``Model``, mirroring
  how ``features`` works — simple, inspectable at definition time.
- Named strings resolve to sklearn-compatible loss functions (the most
  common framework used in Geronimo examples).
- Custom callables are accepted and validated for signature compatibility.
- ``None`` (default) means no objective is declared — existing behavior
  is preserved (backward compatible).

Example:
    ```python
    from geronimo.models import Model, ObjectiveSpec

    class MyModel(Model):
        name = "my-model"
        version = "1.0.0"
        objective = ObjectiveSpec("mse")  # Named objective

        def train(self, X, y, params):
            ...
    ```
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

# Mapping from named strings to sklearn-compatible loss identifiers.
# These are the most common objectives; extend by request.
SUPPORTED_OBJECTIVES: dict[str, str] = {
    "mse": "squared_error",  # sklearn uses "squared_error" for MSE
    "mae": "absolute_error",  # sklearn uses "absolute_error" for MAE
    "binary_crossentropy": "log_loss",
    "categorical_crossentropy": "log_loss",
}


class ObjectiveSpec:
    """Specification for a model's training objective (loss function).

    Supports two modes:
    1. **Named string** — resolves to a known sklearn-compatible loss.
       Raises ``ValueError`` at definition time if the name is unsupported.
    2. **Custom callable** — accepted and validated for signature
       compatibility (must accept ``y_true, y_pred``).

    Args:
        name_or_callable: Either a string key from the supported objectives
            or a callable with signature ``(y_true, y_pred) -> float``.
        description: Optional human-readable description of the objective.

    Raises:
        ValueError: If a named string does not match a supported objective,
            or if a custom callable has an incompatible signature.
    """

    def __init__(
        self,
        name_or_callable: str | Callable[..., float],
        description: str | None = None,
    ) -> None:
        self.description: str | None = description
        self._resolved: str | Callable[..., float] | None = None
        self._type: str = ""

        if isinstance(name_or_callable, str):
            self._validate_named(name_or_callable)
        else:
            self._validate_callable(name_or_callable)

    def _validate_named(self, name: str) -> None:
        """Validate a named objective string."""
        if name not in SUPPORTED_OBJECTIVES:
            supported = sorted(SUPPORTED_OBJECTIVES.keys())
            raise ValueError(
                f"Unsupported objective name: {name!r}. "
                f"Supported objectives for this project: {supported}."
            )
        self._resolved = SUPPORTED_OBJECTIVES[name]
        self._type = "named"
        self._name: str = name

    def _validate_callable(self, fn: Callable[..., float]) -> None:
        """Validate that a callable has a compatible signature."""
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())

        # Must accept at least y_true and y_pred as positional args
        if len(params) < 2:
            raise ValueError(
                f"Custom objective function must accept at least "
                f"two arguments (y_true, y_pred), but {fn.__name__!r} "
                f"has {len(params)} parameter(s): {params}."
            )

        # Check that the first two parameters are positional (not keyword-only)
        first_param = sig.parameters[params[0]]
        second_param = sig.parameters[params[1]]

        # Both must be positional-or-keyword (not keyword-only or var args)
        if first_param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise ValueError(
                f"Custom objective function {fn.__name__!r} must accept "
                f"y_true and y_pred as positional parameters, but the first "
                f"parameter is {first_param.kind.name}."
            )
        if second_param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise ValueError(
                f"Custom objective function {fn.__name__!r} must accept "
                f"y_true and y_pred as positional parameters, but the second "
                f"parameter is {second_param.kind.name}."
            )

        self._resolved = fn
        self._type = "custom"
        self._name = fn.__name__

    @property
    def resolved(self) -> str | Callable[..., float]:
        """The resolved objective: either a string key or a callable."""
        if self._resolved is None:
            raise RuntimeError("ObjectiveSpec has not been initialized.")
        return self._resolved

    @property
    def name(self) -> str:
        """The name of the objective (string name or callable name)."""
        if not self._name:
            return "unnamed"
        return self._name

    @property
    def type(self) -> str:
        """Either 'named' or 'custom'."""
        return self._type

    def __repr__(self) -> str:
        if self._type == "named":
            return f"ObjectiveSpec({self._name!r}, resolved={self._resolved!r})"
        return f"ObjectiveSpec(custom={self._name!r})"
