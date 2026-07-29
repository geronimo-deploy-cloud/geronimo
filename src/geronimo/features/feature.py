"""Feature descriptor for feature definitions."""

import logging
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger(__name__)


class Feature:
    """Feature descriptor for defining individual features.

    Used within FeatureSet classes to define feature columns
    with their types and transformations.

    Order of Operations
    -------------------
    When processing features, the following order is applied:

    1. **derived_feature_fn** (if provided):
       - Called first with the full DataFrame
       - Computes derived values from source_columns
       - Output becomes input for subsequent steps

    2. **transformer** (if provided):
       - Applied after derived_feature_fn (or to source column if no derive fn)
       - Must implement sklearn fit/transform interface
       - Typically for numeric normalization (StandardScaler, MinMaxScaler)

    3. **encoder** (if provided):
       - Applied to categorical values
       - Must implement sklearn fit/transform interface
       - Typically for categorical encoding (OneHotEncoder, LabelEncoder)

    Note: transformer and encoder are mutually exclusive - use one or the other.

    Example:
        ```python
        from geronimo.features import FeatureSet, Feature
        from sklearn.preprocessing import StandardScaler, OneHotEncoder

        class CustomerFeatures(FeatureSet):
            # Simple numeric feature with transformer
            age = Feature(dtype="numeric", transformer=StandardScaler())

            # Categorical feature with encoder
            segment = Feature(dtype="categorical", encoder=OneHotEncoder())

            # Derived feature: single input → custom logic
            age_bucket = Feature(
                dtype="derived",
                source_columns=["age"],
                derived_feature_fn=lambda df: (df["age"] // 10) * 10,
            )

            # Derived feature: multiple inputs → single output
            bmi = Feature(
                dtype="derived",
                source_columns=["weight_kg", "height_m"],
                derived_feature_fn=lambda df: df["weight_kg"] / (df["height_m"] ** 2),
            )

            # Derived + transformed: compute then normalize
            bmi_normalized = Feature(
                dtype="derived",
                source_columns=["weight_kg", "height_m"],
                derived_feature_fn=lambda df: df["weight_kg"] / (df["height_m"] ** 2),
                transformer=StandardScaler(),  # Applied after derive
            )

            # Drop from final output
            name = Feature(dtype="text", drop=True)
        ```
    """

    def __init__(
        self,
        dtype: Literal["numeric", "categorical", "text", "derived"] = "numeric",
        data_type: Optional[type] = None,
        transformer: Optional[Any] = None,
        encoder: Optional[Any] = None,
        source_column: Optional[str] = None,
        source_columns: Optional[list[str]] = None,
        derived_feature_fn: Optional[Callable] = None,
        drop: bool = False,
        description: Optional[str] = None,
        required: bool = True,
        default: Any = None,
    ):
        """Initialize feature.

        Args:
            dtype: Feature data type.
                - "numeric": Numeric values (int, float)
                - "categorical": Categorical/discrete values
                - "text": Text data (typically dropped or embedded)
                - "derived": Computed from other columns via derived_feature_fn

            data_type: Optional Python type for runtime data enforcement.
                When set, fit/transform will validate and coerce incoming data
                to this type. If the data already matches, proceeds silently.
                If coercible, coerces and emits a logging.warning. If not
                coercible, raises a TypeError. If not set, no type checking
                occurs (backward-compatible).
                Examples: float, int, str

            transformer: Sklearn-compatible transformer for numeric features.
                Applied AFTER derived_feature_fn if both are provided.
                Must implement fit() and transform() methods.
                Example: StandardScaler(), MinMaxScaler()

            encoder: Sklearn-compatible encoder for categorical features.
                Must implement fit() and transform() methods.
                Example: OneHotEncoder(), LabelEncoder()

            source_column: Single input column name (if different from attribute name).
                Used when feature maps 1:1 from a differently-named source column.

            source_columns: List of input column names for derived features.
                Required when derived_feature_fn needs multiple input columns.

            derived_feature_fn: Custom function for feature engineering.
                Receives full DataFrame, returns Series or array.
                Called BEFORE transformer (if both provided).
                Example: lambda df: df["weight"] / (df["height"] ** 2)

            drop: If True, exclude feature from final output.
                Useful for passthrough columns needed only for derived features.

            description: Optional human-readable feature description.

            required: Whether the feature must be present in input data.
                If True (default), a missing column raises ValueError. If False,
                missing data is handled by substituting `default` (if set) or
                passing NaN through silently.

            default: Fallback value used when an optional feature's data is
                absent. Only meaningful when `required=False`. When the feature
                is missing and a default is set, the default is substituted and
                a warning is logged.
        """
        self.dtype = dtype
        self.data_type = data_type
        self.transformer = transformer
        self.encoder = encoder
        self.source_column = source_column
        self.source_columns = source_columns
        self.derived_feature_fn = derived_feature_fn
        self.drop = drop
        self.description = description
        self.required = required
        self.default = default
        self._name: Optional[str] = None
    
    dtype: Literal["numeric", "categorical", "text", "derived"]
    """Feature data type."""

    data_type: Optional[type]
    """Optional Python type for runtime data enforcement (e.g. float, int, str)."""

    transformer: Optional[Any]
    """Sklearn-compatible transformer for numeric features."""

    encoder: Optional[Any]
    """Sklearn-compatible encoder for categorical features."""

    source_column: Optional[str]
    """Single input column name."""

    source_columns: Optional[list[str]]
    """List of input column names for derived features."""

    derived_feature_fn: Optional[Callable]
    """Custom function for feature engineering."""

    drop: bool
    """If True, exclude feature from final output."""

    description: Optional[str]
    """Optional human-readable feature description."""

    required: bool
    """Whether the feature must be present in input data."""

    default: Any
    """Fallback value for optional features when data is absent."""

    def __set_name__(self, owner, name: str) -> None:
        """Capture attribute name when defined in class."""
        self._name = name
        if self.source_column is None and self.source_columns is None:
            self.source_column = name

    @property
    def name(self) -> str:
        """Get feature name."""
        return self._name or "unnamed"

    @property
    def input_columns(self) -> list[str]:
        """Get list of input column names."""
        if self.source_columns:
            return self.source_columns
        return [self.source_column or self.name]

    @property
    def has_transformer(self) -> bool:
        """Check if feature has a transformer."""
        return self.transformer is not None

    @property
    def has_encoder(self) -> bool:
        """Check if feature has an encoder."""
        return self.encoder is not None

    @property
    def has_derived_fn(self) -> bool:
        """Check if feature has a derived feature function."""
        return self.derived_feature_fn is not None

    @property
    def is_derived(self) -> bool:
        """Check if feature is derived from custom function."""
        return self.derived_feature_fn is not None or self.dtype == "derived"

    def apply(self, df) -> Any:
        """Apply derived feature function to DataFrame.

        Args:
            df: Input DataFrame with source columns.

        Returns:
            Transformed feature values (Series or array).
        """
        if self.derived_feature_fn is not None:
            return self.derived_feature_fn(df)
        elif self.source_column:
            return df[self.source_column]
        else:
            return df[self.name]

    def check_presence(self, df) -> bool:
        """Check whether this feature's data is present in the DataFrame.

        An absence check that handles required enforcement and default
        substitution. Designed to be called from FeatureSet._process_feature
        before any data processing.

        Args:
            df: Input DataFrame.

        Returns:
            True if the feature's data is present (or was substituted with
            a default). False if the feature is absent and no default was
            substituted (caller should skip this feature).

        Raises:
            ValueError: If the feature is required but absent.
        """
        col_name = self.source_column or self.name

        # Derived features: check source columns
        if self.has_derived_fn:
            sources = self.source_columns or [self.source_column or self.name]
            for src in sources:
                if src not in df.columns:
                    if self.required:
                        raise ValueError(
                            f"Feature '{self.name}' is required but was not found in the input"
                        )
                    if self.default is not None:
                        logger.warning(
                            f"Feature '{self.name}': missing data substituted with default value {self.default}"
                        )
                        # Signal that a default was substituted by returning True
                        # (caller will need to inject the default)
                        return True
                    return False
            return True

        # Standard features: check single column
        if col_name not in df.columns:
            if self.required:
                raise ValueError(
                    f"Feature '{self.name}' is required but was not found in the input"
                )
            if self.default is not None:
                logger.warning(
                    f"Feature '{self.name}': missing data substituted with default value {self.default}"
                )
                return True
            return False

        return True

    def __repr__(self) -> str:
        extras = []
        if self.has_derived_fn:
            extras.append("derived_feature_fn")
        if self.source_columns:
            extras.append(f"inputs={self.source_columns}")
        if self.has_transformer:
            extras.append("transformer")
        if self.has_encoder:
            extras.append("encoder")
        if not self.required:
            extras.append(f"required={self.required}")
        if self.default is not None:
            extras.append(f"default={self.default}")
        if self.data_type is not None:
            extras.append(f"data_type={self.data_type.__name__}")
        extra_str = f", {', '.join(extras)}" if extras else ""
        return f"Feature({self.name}, dtype={self.dtype}{extra_str})"
    def _data_type_matches(self, data) -> bool:
        """Check whether data's actual type matches the declared data_type.

        Maps pandas/numpy dtypes to Python types for comparison.

        Args:
            data: pandas Series, numpy array, or Python iterable.

        Returns:
            True if the data's type matches the declared data_type, False otherwise.
        """
        if self.data_type is None:
            return True

        if hasattr(data, "dtype"):
            dtype_str = str(data.dtype)
            if dtype_str.startswith("float"):
                return self.data_type == float
            elif dtype_str.startswith("int"):
                return self.data_type == int
            elif dtype_str == "object":
                return self.data_type == str
            elif dtype_str == "category":
                return self.data_type == str
            else:
                return self.data_type == data.dtype
        else:
            return type(data) == self.data_type

    def _validate_and_coerce(self, data) -> Any:
        """Validate and coerce data to the declared data_type.

        Behavior:
            - If data_type is None (default): no-op, return data unchanged.
            - If data already matches data_type: no-op, return data unchanged.
            - If data is coercible: coerce and emit a logging.warning.
            - If data is not coercible: raise a TypeError.

        Args:
            data: pandas Series, numpy array, or Python iterable.

        Returns:
            The (possibly coerced) data.

        Raises:
            TypeError: If data cannot be coerced to the declared data_type.
        """
        if self.data_type is None:
            return data

        # If data already matches the declared type, no coercion needed
        if self._data_type_matches(data):
            return data

        # Attempt coercion
        actual_type = self._get_actual_type(data)
        try:
            if hasattr(data, "dtype"):
                # pandas Series or numpy array
                data = data.astype(self.data_type)
            else:
                # Python list or other iterable
                data = [self.data_type(d) for d in data]
        except (TypeError, ValueError, AttributeError) as e:
            raise TypeError(
                f"Feature '{self.name}': expected {self._format_type(self.data_type)}, received {actual_type} and could not coerce"
            ) from e

        # Coercion succeeded — emit warning
        logger.warning(
            f"Feature '{self.name}': coerced data from {actual_type} to {self._format_type(self.data_type)}"
        )
        return data

    @staticmethod
    def _get_actual_type(data) -> str:
        """Get a string representation of data's actual type."""
        if hasattr(data, "dtype"):
            return str(data.dtype)
        return type(data).__name__

    @staticmethod
    def _format_type(t) -> str:
        """Format a type for display in error/warning messages."""
        if hasattr(t, "__name__"):
            return t.__name__
        return str(t)
