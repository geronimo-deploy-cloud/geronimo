"""Feature descriptor for feature definitions."""

from typing import Any, Callable, Literal, Optional


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
        transformer: Optional[Any] = None,
        encoder: Optional[Any] = None,
        source_column: Optional[str] = None,
        source_columns: Optional[list[str]] = None,
        derived_feature_fn: Optional[Callable] = None,
        drop: bool = False,
        description: Optional[str] = None,
    ):
        """Initialize feature.

        Args:
            dtype: Feature data type.
                - "numeric": Numeric values (int, float)
                - "categorical": Categorical/discrete values
                - "text": Text data (typically dropped or embedded)
                - "derived": Computed from other columns via derived_feature_fn

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
        """
        self.dtype = dtype
        self.transformer = transformer
        self.encoder = encoder
        self.source_column = source_column
        self.source_columns = source_columns
        self.derived_feature_fn = derived_feature_fn
        self.drop = drop
        self.description = description
        self._name: Optional[str] = None

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
        extra_str = f", {', '.join(extras)}" if extras else ""
        return f"Feature({self.name}, dtype={self.dtype}{extra_str})"
