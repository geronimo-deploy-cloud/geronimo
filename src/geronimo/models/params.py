"""HyperParameters abstraction for model tuning."""

from typing import Any, Iterator


class HyperParams:
    """Hyperparameter configuration with grid search support.

    Example:
        ```python
        from geronimo.models import HyperParams

        # Define parameter space
        params = HyperParams(
            n_estimators=[100, 200, 500],
            max_depth=[3, 5, 7],
            learning_rate=0.1,  # Fixed value
        )

        # Iterate over grid
        for combo in params.grid():
            model.train(X, y, combo)

        # Get as dict
        params.to_dict()  # Returns first combination
        ```
    """

    def __init__(self, **params):
        """Initialize hyperparameters.

        Args:
            **params: Parameter name-value pairs. Values can be:
                - Single value (fixed parameter)
                - List of values (for grid search)
        """
        self._params = params

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with single values.

        For list parameters, returns the first value.

        Returns:
            Dictionary of parameter values.
        """
        return {
            k: v[0] if isinstance(v, list) else v for k, v in self._params.items()
        }

    def grid(self) -> Iterator["HyperParams"]:
        """Generate all parameter combinations for grid search.

        Yields:
            HyperParams instance for each combination.
        """
        import itertools

        # Separate fixed vs grid parameters
        keys = []
        values = []
        for k, v in self._params.items():
            keys.append(k)
            values.append(v if isinstance(v, list) else [v])

        # Generate cartesian product
        for combo in itertools.product(*values):
            yield HyperParams(**dict(zip(keys, combo)))

    def __getattr__(self, name: str) -> Any:
        """Access parameter by attribute."""
        if name.startswith("_"):
            return super().__getattribute__(name)
        if name in self._params:
            v = self._params[name]
            return v[0] if isinstance(v, list) else v
        raise AttributeError(f"No parameter: {name}")

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self._params.items())
        return f"HyperParams({params_str})"
