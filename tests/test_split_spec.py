"""Tests for SplitSpec and DataSource.split integration."""

import pytest

from geronimo.data_sources import DataSource, SplitSpec


class TestSplitSpec:
    """Tests for SplitSpec class."""

    def test_random_split_reproducibility(self):
        """Same seed + same input data produces identical split across runs."""
        df = DataSource(
            name="test",
            source="function",
            handle=lambda: __import__("pandas").DataFrame(
                {"a": range(100), "b": range(100, 200)}
            ),
        ).load()

        spec = SplitSpec(strategy="random", ratio=0.8, random_seed=42)

        train1, eval1 = spec.split(df)
        train2, eval2 = spec.split(df)

        __import__("pandas").testing.assert_frame_equal(train1, train2)
        __import__("pandas").testing.assert_frame_equal(eval1, eval2)

    def test_random_split_produces_expected_sizes(self):
        """Random split respects the ratio."""
        df = DataSource(
            name="test",
            source="function",
            handle=lambda: __import__("pandas").DataFrame(
                {"x": range(100)}
            ),
        ).load()

        spec = SplitSpec(strategy="random", ratio=0.7, random_seed=0)
        train, eval_ = spec.split(df)

        assert len(train) == 70
        assert len(eval_) == 30
        assert len(df) == 100

    def test_random_split_default_ratio(self):
        """Default ratio is 0.8 (80/20)."""
        df = DataSource(
            name="test",
            source="function",
            handle=lambda: __import__("pandas").DataFrame(
                {"x": range(100)}
            ),
        ).load()

        spec = SplitSpec()  # defaults: strategy="random", ratio=0.8
        train, eval_ = spec.split(df)

        assert len(train) == 80
        assert len(eval_) == 20

    def test_time_split_correctness(self):
        """Time-based split correctly separates rows by cutoff."""
        import pandas as pd

        data = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=365),
            "value": range(365),
        })

        spec = SplitSpec(
            strategy="time",
            datetime_column="date",
            cutoff="2020-06-01",
        )

        train, eval_ = spec.split(data)

        # All training rows should be <= cutoff
        assert train["date"].max() <= pd.Timestamp("2020-06-01")
        # All eval rows should be > cutoff
        assert eval_["date"].min() > pd.Timestamp("2020-06-01")

    def test_time_split_missing_datetime_column(self):
        """Time-based split raises ValueError if datetime column is absent."""
        import pandas as pd

        df = pd.DataFrame({"x": range(10), "y": range(10)})

        spec = SplitSpec(
            strategy="time",
            datetime_column="nonexistent",
            cutoff="2020-01-01",
        )

        with pytest.raises(ValueError, match="not found"):
            spec.split(df)

    def test_time_split_missing_datetime_column_attr(self):
        """Time-based split raises ValueError if datetime_column not set."""
        import pandas as pd

        df = pd.DataFrame({"x": range(10)})

        spec = SplitSpec(strategy="time", cutoff="2020-01-01")
        # datetime_column defaults to None
        with pytest.raises(ValueError, match="requires 'datetime_column'"):
            spec.split(df)

    def test_time_split_missing_cutoff(self):
        """Time-based split raises ValueError if cutoff not set."""
        import pandas as pd

        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10)})

        spec = SplitSpec(strategy="time", datetime_column="date")
        # cutoff defaults to None
        with pytest.raises(ValueError, match="requires 'cutoff'"):
            spec.split(df)

    def test_invalid_strategy(self):
        """Unsupported strategy raises ValueError."""
        import pandas as pd

        df = pd.DataFrame({"x": range(10)})

        spec = SplitSpec(strategy="stratified")
        with pytest.raises(ValueError, match="Unsupported split strategy"):
            spec.split(df)

    def test_invalid_ratio_zero(self):
        """Ratio of 0 raises ValueError."""
        with pytest.raises(ValueError, match="Split ratio must be between"):
            SplitSpec(ratio=0)

    def test_invalid_ratio_negative(self):
        """Negative ratio raises ValueError."""
        with pytest.raises(ValueError, match="Split ratio must be between"):
            SplitSpec(ratio=-0.1)

    def test_invalid_ratio_one(self):
        """Ratio of 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Split ratio must be between"):
            SplitSpec(ratio=1.0)

    def test_invalid_ratio_greater_than_one(self):
        """Ratio > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Split ratio must be between"):
            SplitSpec(ratio=1.5)

    def test_empty_dataframe_raises(self):
        """Splitting an empty DataFrame raises ValueError."""
        import pandas as pd

        df = pd.DataFrame({"x": []})
        spec = SplitSpec()
        with pytest.raises(ValueError, match="empty DataFrame"):
            spec.split(df)

    def test_split_spec_on_data_source(self):
        """SplitSpec can be attached to a DataSource."""
        import pandas as pd

        source = DataSource(
            name="test",
            source="function",
            handle=lambda: pd.DataFrame({"a": range(50), "b": range(50, 100)}),
            split_spec=SplitSpec(strategy="random", ratio=0.6, random_seed=7),
        )

        df = source.load()
        train, eval_ = source.split_spec.split(df)

        assert len(train) == 30
        assert len(eval_) == 20

    def test_data_source_split_convenience_method(self):
        """DataSource.split() loads and splits in one call."""
        import pandas as pd

        source = DataSource(
            name="test",
            source="function",
            handle=lambda: pd.DataFrame({"a": range(50), "b": range(50, 100)}),
            split_spec=SplitSpec(strategy="random", ratio=0.6, random_seed=7),
        )

        train, eval_ = source.split()

        assert len(train) == 30
        assert len(eval_) == 20

    def test_data_source_split_without_split_spec_raises(self):
        """Calling DataSource.split() without a split_spec raises ValueError."""
        source = DataSource(
            name="test",
            source="function",
            handle=lambda: __import__("pandas").DataFrame({"a": range(10)}),
        )

        with pytest.raises(ValueError, match="no split_spec"):
            source.split()


class TestSplitSpecDefaults:
    """Tests for SplitSpec default values."""

    def test_default_strategy_is_random(self):
        """Default strategy is 'random'."""
        spec = SplitSpec()
        assert spec.strategy == "random"

    def test_default_ratio_is_0_8(self):
        """Default ratio is 0.8."""
        spec = SplitSpec()
        assert spec.ratio == 0.8

    def test_default_seed_is_42(self):
        """Default random_seed is 42."""
        spec = SplitSpec()
        assert spec.random_seed == 42

    def test_default_datetime_column_is_none(self):
        """Default datetime_column is None."""
        spec = SplitSpec()
        assert spec.datetime_column is None

    def test_default_cutoff_is_none(self):
        """Default cutoff is None."""
        spec = SplitSpec()
        assert spec.cutoff is None
