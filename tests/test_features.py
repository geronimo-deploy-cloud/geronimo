"""Tests for geronimo.features module."""

import pandas as pd
import pytest

from geronimo.features import Feature, FeatureSet


class TestFeature:
    """Tests for Feature descriptor."""

    def test_basic_feature(self):
        """Test basic feature creation."""
        f = Feature(dtype="numeric")
        assert f.dtype == "numeric"
        assert f.transformer is None
        assert f.drop is False

    def test_feature_with_source_column(self):
        """Test feature with source_column mapping."""
        f = Feature(dtype="numeric", source_column="original_name")
        assert f.source_column == "original_name"

    def test_feature_with_source_columns(self):
        """Test feature with multiple source columns."""
        f = Feature(
            dtype="derived",
            source_columns=["col1", "col2"],
            derived_feature_fn=lambda df: df["col1"] + df["col2"],
        )
        assert f.source_columns == ["col1", "col2"]
        assert f.input_columns == ["col1", "col2"]

    def test_feature_derived_fn(self):
        """Test feature with derived function."""
        f = Feature(
            dtype="derived",
            source_columns=["a", "b"],
            derived_feature_fn=lambda df: df["a"] / df["b"],
        )
        assert f.has_derived_fn is True
        assert f.is_derived is True

    def test_feature_apply(self):
        """Test applying derived function."""
        f = Feature(
            dtype="derived",
            source_columns=["x", "y"],
            derived_feature_fn=lambda df: df["x"] * 2,
        )
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = f.apply(df)
        assert list(result) == [2, 4, 6]

    def test_feature_repr(self):
        """Test feature string representation."""
        f = Feature(dtype="numeric")
        f._name = "age"
        assert "Feature(age" in repr(f)


class TestFeatureSet:
    """Tests for FeatureSet class."""

    def test_basic_feature_set(self, sample_df):
        """Test basic feature set creation and fitting."""
        class SimpleFeatures(FeatureSet):
            age = Feature(dtype="numeric")
            income = Feature(dtype="numeric")

        fs = SimpleFeatures()
        assert len(fs._features) == 2
        assert "age" in fs.feature_names
        assert "income" in fs.feature_names

    def test_feature_set_fit_transform(self, sample_df):
        """Test fit_transform on feature set."""
        class SimpleFeatures(FeatureSet):
            age = Feature(dtype="numeric")
            income = Feature(dtype="numeric")

        fs = SimpleFeatures()
        result = fs.fit_transform(sample_df)
        
        assert fs.is_fitted is True
        assert "age" in result.columns
        assert "income" in result.columns
        assert len(result) == len(sample_df)

    def test_feature_set_with_transformer(self, sample_df):
        """Test feature set with sklearn transformer."""
        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            pytest.skip("sklearn not installed")

        class ScaledFeatures(FeatureSet):
            age = Feature(dtype="numeric", transformer=StandardScaler())

        fs = ScaledFeatures()
        result = fs.fit_transform(sample_df)
        
        # StandardScaler produces mean 0
        assert abs(result["age"].mean()) < 0.01

    def test_feature_set_with_derived(self, sample_df):
        """Test feature set with derived features."""
        class DerivedFeatures(FeatureSet):
            age = Feature(dtype="numeric")
            age_bucket = Feature(
                dtype="derived",
                source_columns=["age"],
                derived_feature_fn=lambda df: (df["age"] // 10) * 10,
            )

        fs = DerivedFeatures()
        result = fs.fit_transform(sample_df)
        
        assert "age_bucket" in result.columns
        assert list(result["age_bucket"]) == [20, 30, 40, 50, 60]

    def test_feature_set_drop(self, sample_df):
        """Test dropping features from output."""
        class FilteredFeatures(FeatureSet):
            age = Feature(dtype="numeric")
            name = Feature(dtype="text", drop=True)

        fs = FilteredFeatures()
        fs.fit(sample_df)
        result = fs.transform(sample_df)
        
        assert "age" in result.columns
        assert "name" not in result.columns

    def test_feature_set_not_fitted_error(self, sample_df):
        """Test error when transforming without fitting."""
        class SimpleFeatures(FeatureSet):
            age = Feature(dtype="numeric")

        fs = SimpleFeatures()
        with pytest.raises(ValueError, match="not fitted"):
            fs.transform(sample_df)
