"""Feature descriptor for feature definitions."""

from typing import Any, Literal, Optional


class Feature:
    """Feature descriptor for defining individual features.

    Used within FeatureSet classes to define feature columns
    with their types and transformations.

    Example:
        ```python
        from geronimo.features import FeatureSet, Feature
        from sklearn.preprocessing import StandardScaler, OneHotEncoder

        class CustomerFeatures(FeatureSet):
            age = Feature(dtype="numeric", transformer=StandardScaler())
            income = Feature(dtype="numeric", transformer=StandardScaler())
            segment = Feature(dtype="categorical", encoder=OneHotEncoder())
            name = Feature(dtype="text", drop=True)  # Excluded from features
        ```
    """

    def __init__(
        self,
        dtype: Literal["numeric", "categorical", "text"] = "numeric",
        transformer: Optional[Any] = None,
        encoder: Optional[Any] = None,
        source_column: Optional[str] = None,
        drop: bool = False,
        description: Optional[str] = None,
    ):
        """Initialize feature.

        Args:
            dtype: Feature data type.
            transformer: Sklearn-compatible transformer for numeric features.
            encoder: Sklearn-compatible encoder for categorical features.
            source_column: Original column name if different from attribute name.
            drop: If True, exclude from output features.
            description: Optional feature description.
        """
        self.dtype = dtype
        self.transformer = transformer
        self.encoder = encoder
        self.source_column = source_column
        self.drop = drop
        self.description = description
        self._name: Optional[str] = None

    def __set_name__(self, owner, name: str) -> None:
        """Capture attribute name when defined in class."""
        self._name = name
        if self.source_column is None:
            self.source_column = name

    @property
    def name(self) -> str:
        """Get feature name."""
        return self._name or "unnamed"

    @property
    def has_transformer(self) -> bool:
        """Check if feature has a transformer."""
        return self.transformer is not None

    @property
    def has_encoder(self) -> bool:
        """Check if feature has an encoder."""
        return self.encoder is not None

    def __repr__(self) -> str:
        return f"Feature({self.name}, dtype={self.dtype})"
