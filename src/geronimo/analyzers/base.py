"""Base classes for project analysis.

This module defines the abstraction layer for analyzers, allowing both
deterministic (rule-based) and AI-powered implementations.

The design supports future GenAI integration by:
1. Using structured dataclasses for analysis results
2. Including confidence scores for uncertain detections
3. Tracking items that need human review
4. Providing hooks for interactive refinement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geronimo.scanners.project import ProjectScan

from geronimo.config.schema import MLFramework


@dataclass
class EndpointInfo:
    """Information about an API endpoint."""

    path: str
    """URL path (e.g., '/v1/predict')."""

    method: str
    """HTTP method (GET, POST, etc.)."""

    handler_function: str
    """Name of the handler function."""

    file_path: Path
    """Path to the file containing the endpoint."""

    request_schema: str | None = None
    """Name of the Pydantic request model, if detected."""

    response_schema: str | None = None
    """Name of the Pydantic response model, if detected."""

    description: str | None = None
    """Docstring or description of the endpoint."""


@dataclass
class ServiceInfo:
    """Information about a service module."""

    name: str
    """Service module name."""

    file_path: Path
    """Path to the service file."""

    functions: list[str] = field(default_factory=list)
    """Public functions in the service."""

    classes: list[str] = field(default_factory=list)
    """Classes defined in the service."""

    dependencies: list[str] = field(default_factory=list)
    """Other modules this service imports from the project."""


@dataclass
class PreprocessingStep:
    """A step in the preprocessing pipeline."""

    name: str
    """Name of the preprocessing function or class."""

    source_file: Path
    """File containing this step."""

    source_function: str
    """Function or method name."""

    order: int = 0
    """Order in the pipeline (0 = first)."""

    input_type: str | None = None
    """Expected input type, if detected."""

    output_type: str | None = None
    """Output type, if detected."""


@dataclass
class ModelArtifactInfo:
    """Information about a model artifact file."""

    path: Path
    """Path to the artifact file."""

    framework: MLFramework | None = None
    """Detected framework for this artifact."""

    size_bytes: int = 0
    """File size in bytes."""

    is_primary: bool = False
    """Whether this appears to be the main model."""


@dataclass
class AnalysisResult:
    """Complete analysis result for a project.

    This structured output is designed to be:
    1. Serializable for caching/logging
    2. Inspectable for debugging
    3. Compatible with both deterministic and AI analysis
    """

    # Core detections
    detected_framework: MLFramework
    """Primary ML framework detected."""

    python_version: str
    """Python version requirement."""

    dependencies: list[str]
    """Project dependencies."""

    # Structure analysis
    endpoints: list[EndpointInfo] = field(default_factory=list)
    """Detected API endpoints."""

    services: list[ServiceInfo] = field(default_factory=list)
    """Detected service modules."""

    preprocessing_chain: list[PreprocessingStep] = field(default_factory=list)
    """Detected preprocessing pipeline."""

    model_artifacts: list[ModelArtifactInfo] = field(default_factory=list)
    """Detected model artifacts."""

    # Quality indicators
    confidence: float = 1.0
    """Overall confidence in the analysis (0-1)."""

    needs_review: list[str] = field(default_factory=list)
    """Items that need human review or clarification."""

    warnings: list[str] = field(default_factory=list)
    """Non-blocking issues or suggestions."""

    # Metadata
    analyzer_type: str = "unknown"
    """Type of analyzer that produced this result."""

    raw_data: dict[str, Any] = field(default_factory=dict)
    """Additional data for debugging or AI context."""


class Analyzer(ABC):
    """Abstract base class for project analyzers.

    This abstraction allows for different analysis implementations:
    - DeterministicAnalyzer: Rule-based, AST parsing
    - GenAIAnalyzer (future): LLM-powered code understanding
    - HybridAnalyzer (future): Deterministic first, AI for uncertain items

    The interface is designed to support:
    1. Single-shot analysis via analyze()
    2. Interactive refinement via refine() (future)
    3. Confidence-based routing between implementations
    """

    @property
    @abstractmethod
    def analyzer_type(self) -> str:
        """Return the type identifier for this analyzer."""
        pass

    @abstractmethod
    def analyze(self, project: "ProjectScan") -> AnalysisResult:
        """Analyze a scanned project.

        Args:
            project: ProjectScan from the scanner.

        Returns:
            AnalysisResult with all detected information.
        """
        pass

    def refine(
        self,
        result: AnalysisResult,
        feedback: dict[str, Any],
    ) -> AnalysisResult:
        """Refine an analysis based on user feedback.

        This is a hook for interactive/conversational refinement.
        Default implementation returns the original result unchanged.

        Args:
            result: Previous analysis result.
            feedback: User-provided corrections or clarifications.

        Returns:
            Updated AnalysisResult.
        """
        return result

    def supports_interactive(self) -> bool:
        """Check if this analyzer supports interactive refinement.

        Returns:
            True if refine() provides meaningful functionality.
        """
        return False
