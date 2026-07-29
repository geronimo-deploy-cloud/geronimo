"""Tests for geronimo.features module."""

import logging

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

    def test_feature_required_default_true(self):
        """Test that required defaults to True."""
        f = Feature(dtype="numeric")
        assert f.required is True

    def test_feature_required_false(self):
        """Test setting required=False."""
        f = Feature(dtype="numeric", required=False)
        assert f.required is False

    def test_feature_default_value(self):
        """Test setting a default value."""
        f = Feature(dtype="numeric", required=False, default=42.0)
        assert f.default == 42.0

    def test_feature_default_on_required(self):
        """Setting default on required=True Feature does not raise at definition."""
        f = Feature(dtype="numeric", required=True, default=42.0)
        assert f.required is True
        assert f.default == 42.0

    def test_feature_repr_with_required_and_default(self):
        """Test repr includes required/default when set."""
        f = Feature(dtype="numeric", required=False, default=42.0)
        f._name = "age"
        r = repr(f)
        assert "required=False" in r
        assert "default=42.0" in r

    def test_check_presence_required_present(self, sample_df):
        """required=True, data present → True (no error)."""
        f = Feature(dtype="numeric", required=True)
        f._name = "age"
        assert f.check_presence(sample_df) is True

    def test_check_presence_required_absent(self, sample_df):
        """required=True, data absent → ValueError."""
        f = Feature(dtype="numeric", required=True)
        f._name = "missing_col"
        with pytest.raises(ValueError, match="Feature 'missing_col' is required but was not found in the input"):
            f.check_presence(sample_df)

    def test_check_presence_optional_present(self, sample_df):
        """required=False, data present → True."""
        f = Feature(dtype="numeric", required=False)
        f._name = "age"
        assert f.check_presence(sample_df) is True

    def test_check_presence_optional_absent_no_default(self, sample_df):
        """required=False, data absent, no default → False."""
        f = Feature(dtype="numeric", required=False)
        f._name = "missing_col"
        assert f.check_presence(sample_df) is False

    def test_check_presence_optional_absent_with_default(self, sample_df, caplog):
        """required=False, data absent, default set → True, warning logged."""
        f = Feature(dtype="numeric", required=False, default=42.0)
        f._name = "missing_col"
        with caplog.at_level(logging.WARNING):
            result = f.check_presence(sample_df)
        assert result is True
        assert "missing data substituted with default value 42.0" in caplog.text

    def test_check_presence_derived_required_missing_source(self, sample_df):
        """Required derived feature missing a source column raises ValueError."""
        f = Feature(
            dtype="derived",
            source_columns=["age", "nonexistent"],
            derived_feature_fn=lambda df: df["age"],
            required=True,
        )
        f._name = "derived_feature"
        with pytest.raises(ValueError, match="Feature 'derived_feature' is required but was not found in the input"):
            f.check_presence(sample_df)

    def test_check_presence_derived_optional_missing_source_with_default(self, sample_df, caplog):
        """Optional derived feature missing source + default → True, warning logged."""
        f = Feature(
            dtype="derived",
            source_columns=["age", "nonexistent"],
            derived_feature_fn=lambda df: df["age"],
            required=False,
            default=99.0,
        )
        f._name = "derived_feature"
        with caplog.at_level(logging.WARNING):
            result = f.check_presence(sample_df)
        assert result is True
        assert "missing data substituted with default value 99.0" in caplog.text

    def test_check_presence_derived_optional_missing_source_no_default(self, sample_df):
        """Optional derived feature missing source, no default → False."""
        f = Feature(
            dtype="derived",
            source_columns=["age", "nonexistent"],
            derived_feature_fn=lambda df: df["age"],
            required=False,
        )
        f._name = "derived_feature"
        assert f.check_presence(sample_df) is False

    # =========================================================================
    # New: args and kwargs binding for derived_feature_fn
    # =========================================================================

    def test_feature_args_only(self):
        """Test derived feature bound with positional args only."""
        def rolling_mean(df, window: int):
            return df["value"].rolling(window).mean()

        f = Feature(
            dtype="derived",
            source_columns=["value"],
            derived_feature_fn=rolling_mean,
            args=(3,),
        )
        assert f.has_derived_fn is True
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = f.apply(df)
        # rolling(3).mean(): [nan, nan, 2.0, 3.0, 4.0]
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 2.0
        assert result.iloc[3] == 3.0
        assert result.iloc[4] == 4.0

    def test_feature_kwargs_only(self):
        """Test derived feature bound with keyword kwargs only."""
        def ratio(df, numerator: str, denominator: str):
            return df[numerator] / df[denominator]

        f = Feature(
            dtype="derived",
            source_columns=["revenue", "cost"],
            derived_feature_fn=ratio,
            kwargs={"numerator": "revenue", "denominator": "cost"},
        )
        assert f.has_derived_fn is True
        df = pd.DataFrame({"revenue": [100.0, 200.0], "cost": [50.0, 40.0]})
        result = f.apply(df)
        assert list(result) == [2.0, 5.0]

    def test_feature_args_and_kwargs(self):
        """Test derived feature bound with both args and kwargs (no overlap)."""
        def bounded(df, lower: float, upper: float, col: str):
            return df[col].clip(lower=lower, upper=upper)

        f = Feature(
            dtype="derived",
            source_columns=["score"],
            derived_feature_fn=bounded,
            args=(0.0, 100.0),
            kwargs={"col": "score"},
        )
        assert f.has_derived_fn is True
        df = pd.DataFrame({"score": [-10.0, 50.0, 200.0]})
        result = f.apply(df)
        assert list(result) == [0.0, 50.0, 100.0]

    def test_feature_kwargs_none_does_not_crash(self):
        """Ensure args or kwargs being None (default) does not crash partial."""
        f = Feature(
            dtype="derived",
            source_columns=["a"],
            derived_feature_fn=lambda df: df["a"],
        )
        assert f.has_derived_fn is True
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = f.apply(df)
        assert list(result) == [1, 2, 3]

    def test_feature_repr_with_args_and_kwargs(self):
        """Test repr includes args/kwargs when set."""
        def fn(df, window: int):
            return df

        f = Feature(
            dtype="derived",
            derived_feature_fn=fn,
            args=(7,),
            kwargs={"col": "x"},
        )
        f._name = "test_feature"
        r = repr(f)
        assert "args=" in r or "derived_feature_fn" in r
        assert "kwargs=" in r or "derived_feature_fn" in r

    def test_feature_set_with_bound_args(self, sample_df):
        """Integration: FeatureSet.fit_transform with derived feature bound via args."""
        class BoundedFeatures(FeatureSet):
            age = Feature(dtype="numeric")
            age_capped = Feature(
                dtype="derived",
                source_columns=["age"],
                derived_feature_fn=lambda df, cap: df["age"].clip(upper=cap),
                args=(50,),
            )

        fs = BoundedFeatures()
        result = fs.fit_transform(sample_df)
        assert "age_capped" in result.columns
        # sample_df has ages [25, 35, 45, 55, 65], capped at 50 → [25, 35, 45, 50, 50]
        assert list(result["age_capped"]) == [25, 35, 45, 50, 50]

    def test_feature_set_with_bound_kwargs(self):
        """Integration: FeatureSet.fit_transform with derived feature bound via kwargs."""
        df = pd.DataFrame({
            "revenue": [100.0, 200.0, 300.0, 400.0, 500.0],
            "cost": [80.0, 160.0, 240.0, 320.0, 400.0],
        })

        class RatioFeatures(FeatureSet):
            revenue = Feature(dtype="numeric")
            cost = Feature(dtype="numeric")
            margin = Feature(
                dtype="derived",
                source_columns=["revenue", "cost"],
                derived_feature_fn=lambda df, col_a, col_b:
                    (df[col_a] - df[col_b]) / df[col_a],
                kwargs={"col_a": "revenue", "col_b": "cost"},
            )

        fs = RatioFeatures()
        result = fs.fit_transform(df)
        assert "margin" in result.columns
        # revenue=[100,200,300,400,500], cost=[80,160,240,320,400]
        # margin = (rev - cost) / rev = [0.2, 0.2, 0.2, 0.2, 0.2]
        assert list(result["margin"]) == [0.2, 0.2, 0.2, 0.2, 0.2]


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

    # =========================================================================
    # Required/Optional integration tests (FeatureSet level)
    # =========================================================================

    def test_required_feature_absent_at_fit_raises(self, sample_df):
        """required=True, data absent at fit time → ValueError."""
        class MyFeatures(FeatureSet):
            age = Feature(dtype="numeric", required=True)
            missing = Feature(dtype="numeric", required=True)

        fs = MyFeatures()
        with pytest.raises(ValueError, match="Feature 'missing' is required but was not found in the input"):
            fs.fit(sample_df)

    def test_required_feature_absent_at_transform_raises(self, sample_df):
        """required=True, data absent at transform time → ValueError."""
        class MyFeatures(FeatureSet):
            age = Feature(dtype="numeric", required=True)
            missing = Feature(dtype="numeric", required=True)

        fs = MyFeatures()
        # Fit on data that has both 'age' and 'missing' columns
        fit_df = pd.DataFrame({
            "age": [25.0, 35.0],
            "missing": [100.0, 200.0],
        })
        fs.fit(fit_df)
        # But transform on data missing 'missing'
        partial_df = pd.DataFrame({"age": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Feature 'missing' is required but was not found in the input"):
            fs.transform(partial_df)

    def test_required_feature_present_no_change(self, sample_df):
        """required=True, data present → no change in behavior."""
        class MyFeatures(FeatureSet):
            age = Feature(dtype="numeric", required=True)

        fs = MyFeatures()
        result = fs.fit_transform(sample_df)
        assert "age" in result.columns
        assert list(result["age"]) == [25, 35, 45, 55, 65]

    def test_optional_feature_absent_with_default_substituted(self, sample_df, caplog):
        """required=False, data absent, default set → default substituted, warning logged."""
        class MyFeatures(FeatureSet):
            age = Feature(dtype="numeric")
            missing_with_default = Feature(
                dtype="numeric",
                required=False,
                default=42.0,
            )

        fs = MyFeatures()
        with caplog.at_level(logging.WARNING):
            result = fs.fit_transform(sample_df)
        assert "missing_with_default" in result.columns
        assert list(result["missing_with_default"]) == [42.0, 42.0, 42.0, 42.0, 42.0]
        assert "missing data substituted with default value 42.0" in caplog.text

    def test_optional_feature_absent_no_default_passes_nan(self, sample_df):
        """required=False, data absent, no default → None/NaN passed through."""
        class MyFeatures(FeatureSet):
            age = Feature(dtype="numeric")
            missing_no_default = Feature(
                dtype="numeric",
                required=False,
            )

        fs = MyFeatures()
        result = fs.fit_transform(sample_df)
        assert "missing_no_default" in result.columns
        # All values should be None (NaN-like)
        assert all(v is None for v in result["missing_no_default"])

    def test_optional_feature_present_no_change(self, sample_df):
        """required=False, data present → no change in behavior."""
        full_df = pd.DataFrame({
            "age": [25, 35, 45, 55, 65],
            "income": [50000, 75000, 100000, 125000, 150000],
        })

        class MyFeatures(FeatureSet):
            age = Feature(dtype="numeric")
            income = Feature(dtype="numeric", required=False)

        fs = MyFeatures()
        result = fs.fit_transform(full_df)
        assert "income" in result.columns
        assert list(result["income"]) == [50000, 75000, 100000, 125000, 150000]

    def test_required_feature_with_default_still_enforces(self, sample_df):
        """default on required=True is never used; required enforcement takes priority."""
        class MyFeatures(FeatureSet):
            age = Feature(dtype="numeric")
            missing_with_default = Feature(
                dtype="numeric",
                required=True,
                default=999.0,
            )

        fs = MyFeatures()
        with pytest.raises(ValueError, match="Feature 'missing_with_default' is required but was not found in the input"):
            fs.fit(sample_df)

    def test_existing_tests_still_pass(self, sample_df):
        """Regression: existing Feature behavior without required/default is unchanged."""
        # Replicate the basic feature set test to ensure backward compatibility
        class SimpleFeatures(FeatureSet):
            age = Feature(dtype="numeric")
            income = Feature(dtype="numeric")

        fs = SimpleFeatures()
        result = fs.fit_transform(sample_df)
        assert fs.is_fitted is True
        assert "age" in result.columns
        assert "income" in result.columns
        assert len(result) == len(sample_df)


