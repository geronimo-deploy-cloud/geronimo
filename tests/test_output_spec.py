"""Tests for geronimo.batch.output_spec module."""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from geronimo.batch import BatchPipeline, OutputSpec
from geronimo.batch.output_spec import (
    Destination,
    Format,
    _write_local,
    _write_output,
    _write_s3,
)


def _make_mock_boto3():
    """Return a mock boto3 module with an S3 client."""
    mock_s3 = MagicMock()
    mock_s3.upload_file = MagicMock()
    mock_boto3 = MagicMock()
    mock_boto3.client = MagicMock(return_value=mock_s3)
    return mock_boto3, mock_s3


class TestOutputSpec:
    """Tests for OutputSpec class."""

    def test_local_csv_spec(self):
        """Test creating a local CSV output spec."""
        spec = OutputSpec(
            destination="local",
            path="./outputs",
            format="csv",
        )
        assert spec.destination == Destination.LOCAL
        assert spec.format == Format.CSV
        assert spec.path == "./outputs"
        assert spec.filename is None

    def test_local_parquet_spec(self):
        """Test creating a local Parquet output spec."""
        spec = OutputSpec(
            destination="local",
            path="/data/results",
            format="parquet",
        )
        assert spec.destination == Destination.LOCAL
        assert spec.format == Format.PARQUET

    def test_s3_parquet_spec(self):
        """Test creating an S3 Parquet output spec."""
        spec = OutputSpec(
            destination="s3",
            path="s3://my-bucket/outputs/",
            format="parquet",
        )
        assert spec.destination == Destination.S3
        assert spec.format == Format.PARQUET

    def test_s3_csv_spec(self):
        """Test creating an S3 CSV output spec."""
        spec = OutputSpec(
            destination="s3",
            path="s3://bucket/data/",
            format="csv",
        )
        assert spec.destination == Destination.S3
        assert spec.format == Format.CSV

    def test_custom_filename(self):
        """Test OutputSpec with custom filename."""
        spec = OutputSpec(
            destination="local",
            path="./outputs",
            format="csv",
            filename="predictions",
        )
        assert spec.filename == "predictions"

    def test_s3_path_must_start_with_s3(self):
        """Test that S3 destination requires s3:// prefix."""
        with pytest.raises(ValueError, match="S3 destination requires path starting with"):
            OutputSpec(destination="s3", path="my-bucket/outputs/")

    def test_local_path_must_be_absolute_or_relative(self):
        """Test that local destination requires a valid path."""
        with pytest.raises(ValueError, match="Local destination requires"):
            OutputSpec(destination="local", path="just_a_name")

    def test_enum_inputs(self):
        """Test that Destination and Format enums are accepted."""
        spec = OutputSpec(
            destination=Destination.S3,
            path="s3://bucket/",
            format=Format.CSV,
        )
        assert spec.destination == Destination.S3
        assert spec.format == Format.CSV

    def test_repr(self):
        """Test OutputSpec repr."""
        spec = OutputSpec(
            destination="local",
            path="./out",
            format="csv",
            filename="test",
        )
        assert "OutputSpec" in repr(spec)
        assert "local" in repr(spec)
        assert "csv" in repr(spec)

    def test_build_output_path_local(self):
        """Test building local output path."""
        spec = OutputSpec(
            destination="local",
            path="./outputs",
            format="csv",
            filename="predictions",
        )
        assert spec._build_output_path("MyPipeline") == "./outputs/predictions.csv"

    def test_build_output_path_local_default_filename(self):
        """Test building local path with default filename."""
        spec = OutputSpec(
            destination="local",
            path="./outputs",
            format="parquet",
        )
        assert spec._build_output_path("MyPipeline") == "./outputs/MyPipeline.parquet"

    def test_build_output_path_s3(self):
        """Test building S3 output path."""
        spec = OutputSpec(
            destination="s3",
            path="s3://bucket/outputs/",
            format="csv",
            filename="results",
        )
        assert spec._build_output_path("MyPipeline") == "s3://bucket/outputs/results.csv"

    def test_build_output_path_s3_no_trailing_slash(self):
        """Test S3 path without trailing slash gets added."""
        spec = OutputSpec(
            destination="s3",
            path="s3://bucket/outputs",
            format="parquet",
            filename="results",
        )
        assert spec._build_output_path("MyPipeline") == "s3://bucket/outputs/results.parquet"


class TestWriteLocal:
    """Tests for _write_local."""

    def test_write_csv(self, tmp_path):
        """Test writing a DataFrame as CSV."""
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        output_path = str(tmp_path / "test.csv")
        _write_local(df, output_path, Format.CSV)
        
        result = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(result, df.reset_index(drop=True))

    def test_write_parquet(self, tmp_path):
        """Test writing a DataFrame as Parquet."""
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not installed")
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        output_path = str(tmp_path / "test.parquet")
        _write_local(df, output_path, Format.PARQUET)
        
        result = pd.read_parquet(output_path)
        # Parquet may reorder columns, so compare values
        assert set(result.columns) == {"a", "b"}
        assert len(result) == 2

    def test_write_non_dataframe(self, tmp_path):
        """Test writing non-DataFrame results as JSON."""
        data = {"status": "ok", "count": 42}
        output_path = str(tmp_path / "test.json")
        _write_local(data, output_path, Format.CSV)
        
        loaded = json.loads(open(output_path).read())
        assert loaded == {"status": "ok", "count": 42}

    def test_write_local_unreachable_directory(self):
        """Test error when output directory is unreachable."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(RuntimeError, match="Cannot create output directory"):
            _write_local(df, "/root/no/such/dir/output.csv", Format.CSV)


class TestWriteS3:
    """Tests for _write_s3."""

    def test_write_s3_csv(self):
        """Test writing DataFrame as CSV to S3 (mocked)."""
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        mock_boto3, mock_s3 = _make_mock_boto3()

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            _write_s3(df, "s3://my-bucket/outputs/test.csv", Format.CSV)

        mock_s3.upload_file.assert_called_once()
        call_kwargs = mock_s3.upload_file.call_args
        assert call_kwargs[0][0].endswith(".csv")
        assert call_kwargs[0][1] == "my-bucket"
        assert call_kwargs[0][2] == "outputs/test.csv"

    def test_write_s3_parquet(self):
        """Test writing DataFrame as Parquet to S3 (mocked)."""
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not installed")
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        mock_boto3, mock_s3 = _make_mock_boto3()

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            _write_s3(df, "s3://my-bucket/outputs/test.parquet", Format.PARQUET)

        mock_s3.upload_file.assert_called_once()
        call_kwargs = mock_s3.upload_file.call_args
        assert call_kwargs[0][0].endswith(".parquet")
        assert call_kwargs[0][1] == "my-bucket"
        assert call_kwargs[0][2] == "outputs/test.parquet"

    def test_write_s3_non_dataframe(self):
        """Test that S3 output requires a DataFrame."""
        data = {"status": "ok"}
        with pytest.raises(RuntimeError, match="S3 output requires a pandas DataFrame"):
            _write_s3(data, "s3://bucket/test.csv", Format.CSV)

    def test_write_s3_invalid_uri(self):
        """Test error for invalid S3 URI."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(RuntimeError, match="S3 output path must start with"):
            _write_s3(df, "bucket/test.csv", Format.CSV)

    def test_write_s3_unreachable(self):
        """Test error when S3 is unreachable."""
        df = pd.DataFrame({"a": [1]})
        mock_boto3, mock_s3 = _make_mock_boto3()
        mock_s3.upload_file.side_effect = Exception("Connection refused")

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            with pytest.raises(RuntimeError, match="Failed to write output to S3"):
                _write_s3(df, "s3://bucket/test.csv", Format.CSV)

    def test_write_s3_boto3_not_installed(self):
        """Test error when boto3 is not installed."""
        df = pd.DataFrame({"a": [1]})
        # Remove boto3 from sys.modules if present, then patch the import
        boto3_was_present = "boto3" in sys.modules
        if boto3_was_present:
            del sys.modules["boto3"]
        # Remove any cached boto3 references
        for mod in list(sys.modules.keys()):
            if mod == "boto3" or mod.startswith("boto3."):
                del sys.modules[mod]
        
        with pytest.raises(RuntimeError, match="S3 output requires the 'boto3'"):
            _write_s3(df, "s3://bucket/test.csv", Format.CSV)
        
        # Restore boto3 if it was present
        if boto3_was_present:
            import boto3  # noqa: F401
            sys.modules["boto3"] = boto3


class TestWriteOutput:
    """Tests for the _write_output dispatcher."""

    def test_write_output_local(self, tmp_path):
        """Test _write_output dispatches to local writer."""
        df = pd.DataFrame({"a": [1, 2]})
        spec = OutputSpec(
            destination="local",
            path=str(tmp_path),
            format="csv",
            filename="test",
        )
        result = _write_output(df, spec, "MyPipeline")
        assert result.endswith("test.csv")
        assert os.path.exists(result)

    def test_write_output_s3(self):
        """Test _write_output dispatches to S3 writer."""
        df = pd.DataFrame({"a": [1, 2]})
        spec = OutputSpec(
            destination="s3",
            path="s3://bucket/outputs/",
            format="csv",
            filename="test",
        )
        mock_boto3, mock_s3 = _make_mock_boto3()

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            result = _write_output(df, spec, "MyPipeline")

        assert result == "s3://bucket/outputs/test.csv"
        mock_s3.upload_file.assert_called_once()

    def test_write_output_unsupported_destination(self):
        """Test error for unsupported destination type."""
        class FakeSpec:
            destination = MagicMock()
            destination.value = "snowflake"
            filename = None

            def _build_output_path(self, class_name):
                return f"snowflake://{class_name}.csv"

        with pytest.raises(RuntimeError, match="Unsupported destination"):
            _write_output("data", FakeSpec(), "MyPipeline")


class TestBatchPipelineOutputSpec:
    """Tests for BatchPipeline integration with OutputSpec."""

    def test_pipeline_with_no_output_spec_raises(self):
        """Test that a pipeline without OutputSpec raises at execute()."""
        class TestPipeline(BatchPipeline):
            def run(self):
                return {"status": "ok"}

        pipeline = TestPipeline()
        pipeline._is_initialized = True  # skip full initialization

        with pytest.raises(RuntimeError, match="No OutputSpec configured"):
            pipeline.execute()

    def test_pipeline_with_output_spec_succeeds(self, tmp_path):
        """Test that a pipeline with OutputSpec writes results."""
        class TestPipeline(BatchPipeline):
            output = OutputSpec(
                destination="local",
                path=str(tmp_path),
                format="csv",
                filename="results",
            )

            def run(self):
                return pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        pipeline = TestPipeline()
        pipeline._is_initialized = True  # skip full initialization
        result = pipeline.execute()

        # Check file was written
        files = list(tmp_path.glob("*.csv"))
        assert len(files) == 1
        assert "results.csv" in str(files[0])

    def test_pipeline_with_s3_output_spec(self):
        """Test pipeline with S3 output spec (mocked)."""
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not installed")
        
        class TestPipeline(BatchPipeline):
            output = OutputSpec(
                destination="s3",
                path="s3://my-bucket/outputs/",
                format="parquet",
                filename="predictions",
            )

            def run(self):
                return pd.DataFrame({"a": [1, 2]})

        pipeline = TestPipeline()
        pipeline._is_initialized = True
        mock_boto3, mock_s3 = _make_mock_boto3()

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            result = pipeline.execute()

        mock_s3.upload_file.assert_called_once()
        call_kwargs = mock_s3.upload_file.call_args
        assert call_kwargs[0][2] == "outputs/predictions.parquet"

    def test_output_spec_includes_helpful_error_message(self):
        """Test that the error message for missing OutputSpec is helpful."""
        class TestPipeline(BatchPipeline):
            def run(self):
                return "data"

        pipeline = TestPipeline()
        pipeline._is_initialized = True

        try:
            pipeline.execute()
            pytest.fail("Expected RuntimeError")
        except RuntimeError as e:
            error_msg = str(e)
            assert "OutputSpec" in error_msg
            assert "destination='local'" in error_msg
            assert "path='./outputs'" in error_msg
            assert "format='parquet'" in error_msg

