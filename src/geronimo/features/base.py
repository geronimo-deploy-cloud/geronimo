"""FeatureSet base class for feature engineering."""

from typing import Any, Optional, TYPE_CHECKING

import pandas as pd

from geronimo.features.feature import Feature

if TYPE_CHECKING:
    from geronimo.artifacts import ArtifactStore
    from geronimo.data import DataSource


class FeatureSet:
    """Base class for feature engineering pipelines.

    Provides fit/transform semantics for training vs production,
    with integrated artifact storage for encoders and transformers.

    Example:
        ```python
        from geronimo.features import FeatureSet, Feature
        from sklearn.preprocessing import StandardScaler, OneHotEncoder

        class CustomerFeatures(FeatureSet):
            data_source = DataSource(
                name="customers",
                source="snowflake",
                query=Query.from_file("queries/customers.sql"),
            )

            age = Feature(dtype="numeric", transformer=StandardScaler())
            income = Feature(dtype="numeric", transformer=StandardScaler())
            segment = Feature(dtype="categorical", encoder=OneHotEncoder(sparse_output=False))

        # Training: fit and transform
        features = CustomerFeatures()
        X = features.fit_transform(training_df)

        # Production: transform only (uses fitted encoders)
        X = features.transform(production_df)
        ```
    """

    # Override in subclass
    data_source: Optional["DataSource"] = None

    def __init__(self):
        """Initialize feature set."""
        self._features: dict[str, Feature] = {}
        self._is_fitted: bool = False

        # Collect Feature descriptors from class
        for name in dir(self.__class__):
            attr = getattr(self.__class__, name, None)
            if isinstance(attr, Feature):
                self._features[name] = attr

    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names (excluding dropped)."""
        return [f.name for f in self._features.values() if not f.drop]

    @property
    def numeric_features(self) -> list[Feature]:
        """Get numeric features."""
        return [f for f in self._features.values() if f.dtype == "numeric" and not f.drop]

    @property
    def categorical_features(self) -> list[Feature]:
        """Get categorical features."""
        return [
            f for f in self._features.values() if f.dtype == "categorical" and not f.drop
        ]

    def fit(self, df: pd.DataFrame) -> "FeatureSet":
        """Fit all transformers and encoders.

        Args:
            df: Training DataFrame.

        Returns:
            Self for chaining.
        """
        for feature in self._features.values():
            if feature.drop:
                continue
            self._process_feature(feature, df, mode="fit")

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame using fitted transformers.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame.

        Raises:
            ValueError: If not fitted.
        """
        if not self._is_fitted:
            raise ValueError("FeatureSet not fitted. Call fit() first.")

        result = pd.DataFrame(index=df.index)

        for feature in self._features.values():
            if feature.drop:
                continue
            
            transformed = self._process_feature(feature, df, mode="transform")
            if transformed is not None:
                if isinstance(transformed, dict):
                    # Multi-column output from encoders
                    for col_name, values in transformed.items():
                        result[col_name] = values
                else:
                    result[feature.name] = transformed

        return result

    def _process_feature(
        self,
        feature: Feature,
        df: pd.DataFrame,
        mode: str,
    ) -> any:
        """Process a single feature for fit or transform.
        
        Unified processing logic to reduce code duplication between
        fit() and transform() methods.
        
        Args:
            feature: Feature descriptor to process.
            df: Input DataFrame.
            mode: Either "fit" or "transform".
            
        Returns:
            For mode="fit": None (modifies transformers/encoders in place).
            For mode="transform": Transformed values (Series, array, or dict for multi-column).
        """
        # Handle derived features with custom functions
        if feature.has_derived_fn:
            derived_values = feature.apply(df)
            
            if feature.has_transformer:
                if mode == "fit":
                    feature.transformer.fit(derived_values.values.reshape(-1, 1))
                    return None
                else:  # transform
                    transformed = feature.transformer.transform(
                        derived_values.values.reshape(-1, 1)
                    )
                    return transformed.flatten()
            else:
                if mode == "fit":
                    return None
                return derived_values.values

        # Standard features
        col_name = feature.source_column
        if col_name not in df.columns:
            return None

        if feature.has_transformer:
            if mode == "fit":
                feature.transformer.fit(df[[col_name]])
                return None
            else:  # transform
                transformed = feature.transformer.transform(df[[col_name]])
                return transformed.flatten()
        elif feature.has_encoder:
            if mode == "fit":
                feature.encoder.fit(df[[col_name]])
                return None
            else:  # transform
                encoded = feature.encoder.transform(df[[col_name]])
                # Handle multi-column output from encoders
                if hasattr(feature.encoder, "get_feature_names_out"):
                    enc_names = feature.encoder.get_feature_names_out([col_name])
                    return {enc_name: encoded[:, i] for i, enc_name in enumerate(enc_names)}
                else:
                    return encoded.flatten()
        else:
            if mode == "fit":
                return None
            return df[col_name].values

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Training DataFrame.

        Returns:
            Transformed DataFrame.
        """
        return self.fit(df).transform(df)

    def save(self, store: "ArtifactStore") -> None:
        """Save fitted transformers and encoders to artifact store.

        Args:
            store: ArtifactStore instance.
        """
        for name, feature in self._features.items():
            if feature.has_transformer:
                store.save(
                    f"transformer_{name}",
                    feature.transformer,
                    artifact_type="transformer",
                )
            if feature.has_encoder:
                store.save(
                    f"encoder_{name}",
                    feature.encoder,
                    artifact_type="encoder",
                )

    def load(self, store: "ArtifactStore") -> None:
        """Load fitted transformers and encoders from artifact store.

        Args:
            store: ArtifactStore instance.
        """
        for name, feature in self._features.items():
            if feature.has_transformer:
                feature.transformer = store.get(f"transformer_{name}")
            if feature.has_encoder:
                feature.encoder = store.get(f"encoder_{name}")

        self._is_fitted = True

    @property
    def is_fitted(self) -> bool:
        """Check if feature set has been fitted."""
        return self._is_fitted

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}({len(self._features)} features, {status})"
