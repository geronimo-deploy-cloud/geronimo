"""Output specification for batch pipeline results.

Defines where and in what format batch pipeline outputs should be written.

Design Rationale:
    OutputSpec attaches as the ``output`` attribute on BatchPipeline, consistent
    with ``schedule`` and ``trigger`` being attribute-style declarations rather
    than standalone objects passed into run(). This keeps the pipeline
class-level declaration self-contained and inspectable before execution.

Example:
    ```python
    from geronimo.batch import BatchPipeline, OutputSpec

    class MyPipeline(BatchPipeline):
        output = OutputSpec(
            destination="s3",
            path="s3://my-bucket/outputs/",
            format="parquet",
        )

        def run(self):
            return self.model.predict(self.X)
    ```
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from geronimo.batch.pipeline import BatchPipeline


class Destination(str, Enum):
    """Supported output destinations for batch pipelines."""

    LOCAL = "local"
    S3 = "s3"


class Format(str, Enum):
    """Supported output formats for batch pipelines."""

    CSV = "csv"
    PARQUET = "parquet"


class OutputSpec:
    """Specification for where and how batch pipeline outputs are written.

    Attributes:
        destination: Where to write outputs (local filesystem or S3).
        path: File path or S3 URI prefix for the output.
        format: File format (CSV or Parquet).
        filename: Optional filename (without extension). Defaults to the
            pipeline class name. The extension is derived from the format.

    Raises:
        ValueError: If configuration is invalid (e.g., S3 destination
            without a path starting with ``s3://``).
    """

    destination: Destination
    path: str
    format: Format
    filename: Optional[str]

    def __init__(
        self,
        destination: str | Destination,
        path: str,
        format: str | Format = Format.PARQUET,
        filename: Optional[str] = None,
    ) -> None:
        """Initialize output specification.

        Args:
            destination: Where to write outputs. Must be ``"local"`` or
                ``"s3"`` (or the corresponding ``Destination`` enum value).
            path: File path for local output, or S3 bucket prefix for S3
                output. For S3, this should be a path like
                ``"s3://my-bucket/outputs/"``.
            format: Output file format. Must be ``"csv"`` or ``"parquet"``
                (or the corresponding ``Format`` enum value). Defaults to
                ``"parquet"``.
            filename: Optional base filename (without extension). If not
                provided, the pipeline class name is used. The file extension
                is derived from the format.
        """
        self.destination = Destination(destination)
        self.format = Format(format)
        self.path = path
        self.filename = filename

        # Validate S3 path format
        if self.destination == Destination.S3 and not self.path.startswith("s3://"):
            raise ValueError(
                f"S3 destination requires path starting with 's3://', "
                f"got: {self.path!r}"
            )

        # Validate local path format
        if self.destination == Destination.LOCAL and not (
            self.path.startswith("/") or self.path.startswith("./")
        ):
            raise ValueError(
                f"Local destination requires an absolute or relative path, "
                f"got: {self.path!r}"
            )

    def _build_output_path(self, class_name: str) -> str:
        """Build the full output path with correct extension.

        Args:
            class_name: Pipeline class name used as default filename.

        Returns:
            Full path with format-appropriate extension.
        """
        base = self.filename or class_name
        ext = self._extension()
        if self.destination == Destination.S3:
            # Ensure path ends with /
            prefix = self.path if self.path.endswith("/") else self.path + "/"
            return f"{prefix}{base}.{ext}"
        else:
            return f"{self.path}/{base}.{ext}"

    def _extension(self) -> str:
        """Get the file extension for this format."""
        return "csv" if self.format == Format.CSV else "parquet"

    def __repr__(self) -> str:
        return (
            f"OutputSpec(destination={self.destination.value}, "
            f"path={self.path!r}, format={self.format.value}, "
            f"filename={self.filename!r})"
        )


def _write_output(
    results: Any,
    output_spec: OutputSpec,
    class_name: str,
) -> str:
    """Write pipeline results according to the OutputSpec.

    Args:
        results: The pipeline results (DataFrame, dict, etc.).
        output_spec: The output specification.
        class_name: Pipeline class name for default filename.

    Returns:
        Path or URI where results were saved.

    Raises:
        RuntimeError: If the destination is unreachable or writing fails.
    """
    import json
    from datetime import datetime
    from pathlib import Path

    output_path = output_spec._build_output_path(class_name)

    if output_spec.destination == Destination.LOCAL:
        _write_local(results, output_path, output_spec.format)
    elif output_spec.destination == Destination.S3:
        _write_s3(results, output_path, output_spec.format)
    else:
        raise RuntimeError(
            f"Unsupported destination type: {output_spec.destination.value}. "
            f"Supported destinations: {', '.join(d.value for d in Destination)}"
        )

    return output_path


def _write_local(
    results: Any,
    output_path: str,
    output_format: Format,
) -> None:
    """Write results to local filesystem.

    Args:
        results: The pipeline results.
        output_path: Full path with extension.
        output_format: File format.

    Raises:
        RuntimeError: If the directory cannot be created or writing fails.
    """
    import json
    import pandas as pd
    from pathlib import Path

    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Cannot create output directory '{path.parent}': {e}"
        ) from e

    try:
        if isinstance(results, pd.DataFrame):
            if output_format == Format.CSV:
                results.to_csv(path, index=False)
            else:
                results.to_parquet(path)
        else:
            # Non-DataFrame results: always use JSON
            path = path.with_suffix(".json")
            path.write_text(json.dumps(results, default=str))
    except Exception as e:
        raise RuntimeError(
            f"Failed to write output to local path '{path}': {e}"
        ) from e


def _write_s3(
    results: Any,
    output_path: str,
    output_format: Format,
) -> None:
    """Write results to S3.

    Args:
        results: The pipeline results (must be a pandas DataFrame).
        output_path: Full S3 URI with extension.
        output_format: File format.

    Raises:
        RuntimeError: If S3 writing fails or results are not a DataFrame.
    """
    import pandas as pd
    import tempfile

    if not isinstance(results, pd.DataFrame):
        raise RuntimeError(
            f"S3 output requires a pandas DataFrame, got {type(results).__name__}. "
            f"Wrap non-DataFrame results in a single-row DataFrame."
        )

    # Parse S3 URI
    if not output_path.startswith("s3://"):
        raise RuntimeError(
            f"S3 output path must start with 's3://', got: {output_path!r}"
        )

    path_parts = output_path.replace("s3://", "", 1).split("/", 1)
    if len(path_parts) < 2:
        raise RuntimeError(
            f"Invalid S3 URI '{output_path}': expected 's3://bucket/path/file.ext'"
        )
    bucket = path_parts[0]
    key = path_parts[1]

    try:
        import boto3
    except ImportError as e:
        raise RuntimeError(
            "S3 output requires the 'boto3' package. Install it with: "
            "pip install boto3"
        ) from e

    # Write DataFrame to temp file in the correct format
    ext = "csv" if output_format == Format.CSV else "parquet"
    with tempfile.NamedTemporaryFile(
        suffix=f".{ext}", delete=False
    ) as f:
        temp_path = f.name

    try:
        if output_format == Format.CSV:
            results.to_csv(temp_path, index=False)
        else:
            results.to_parquet(temp_path)

        s3 = boto3.client("s3")
        s3.upload_file(temp_path, bucket, key)
    except Exception as e:
        raise RuntimeError(
            f"Failed to write output to S3 (bucket={bucket}, key={key}): {e}"
        ) from e
    finally:
        import os

        if os.path.exists(temp_path):
            os.unlink(temp_path)
