"""SDK Wrapper Generator for importing existing projects.

Detects patterns in existing code and generates Geronimo SDK wrappers
with TODO tags for items requiring manual configuration.
"""

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class TodoPriority(str, Enum):
    """Priority levels for TODO items."""

    HIGH = "HIGH"      # Blocks functionality
    MEDIUM = "MEDIUM"  # Important but not blocking
    LOW = "LOW"        # Nice to have


@dataclass
class TodoItem:
    """Represents a TODO item requiring manual configuration."""

    message: str
    priority: TodoPriority
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    category: str = "general"

    def to_comment(self) -> str:
        """Generate TODO comment string."""
        location = ""
        if self.file_path:
            location = f" ({self.file_path}"
            if self.line_number:
                location += f":{self.line_number}"
            location += ")"
        return f"# TODO({self.priority.value}): [{self.category}] {self.message}{location}"


@dataclass
class DetectedPattern:
    """A detected pattern that can be wrapped in SDK."""

    pattern_type: str  # "transformer", "encoder", "query", "endpoint", "data_source"
    name: str
    source_file: str
    source_line: int
    generated_code: str
    todos: list[TodoItem] = field(default_factory=list)


@dataclass
class SDKWrapperResult:
    """Result of SDK wrapper generation."""

    detected_patterns: list[DetectedPattern]
    feature_set_code: str
    data_sources_code: str
    model_code: str
    endpoint_code: str
    todos: list[TodoItem]


class SDKWrapperGenerator:
    """Generates Geronimo SDK wrappers from existing project code.

    Detects:
    - sklearn transformers/encoders → Feature descriptors
    - SQL files → DataSource with Query.from_file()
    - FastAPI endpoints → Endpoint class wrappers

    Generates TODO tags for:
    - Custom preprocessing logic
    - Non-standard patterns
    - Missing configurations
    """

    # Known sklearn transformers
    SKLEARN_TRANSFORMERS = {
        "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler",
        "Normalizer", "Binarizer", "QuantileTransformer", "PowerTransformer",
    }

    # Known sklearn encoders
    SKLEARN_ENCODERS = {
        "OneHotEncoder", "LabelEncoder", "OrdinalEncoder", "LabelBinarizer",
        "MultiLabelBinarizer",
    }

    def __init__(self, project_path: Path):
        """Initialize generator.

        Args:
            project_path: Root path of the project.
        """
        self.project_path = project_path
        self.detected_patterns: list[DetectedPattern] = []
        self.todos: list[TodoItem] = []

    def analyze(self) -> SDKWrapperResult:
        """Analyze project and generate SDK wrappers.

        Returns:
            SDKWrapperResult with generated code and TODOs.
        """
        # Detect patterns
        self._detect_sklearn_patterns()
        self._detect_sql_queries()
        self._detect_data_files()

        # Generate wrapper code
        feature_set_code = self._generate_feature_set()
        data_sources_code = self._generate_data_sources()
        model_code = self._generate_model()
        endpoint_code = self._generate_endpoint()

        return SDKWrapperResult(
            detected_patterns=self.detected_patterns,
            feature_set_code=feature_set_code,
            data_sources_code=data_sources_code,
            model_code=model_code,
            endpoint_code=endpoint_code,
            todos=self.todos,
        )

    def _detect_sklearn_patterns(self) -> None:
        """Detect sklearn transformer/encoder usage in Python files."""
        for py_file in self.project_path.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text()
                tree = ast.parse(content)
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    self._check_sklearn_call(node, py_file, content)

    def _check_sklearn_call(self, node: ast.Call, file_path: Path, content: str) -> None:
        """Check if a call is a sklearn transformer/encoder."""
        func_name = self._get_call_name(node)
        if not func_name:
            return

        if func_name in self.SKLEARN_TRANSFORMERS:
            pattern = DetectedPattern(
                pattern_type="transformer",
                name=func_name,
                source_file=str(file_path.relative_to(self.project_path)),
                source_line=node.lineno,
                generated_code=f"Feature(dtype='numeric', transformer={func_name}())",
            )
            self.detected_patterns.append(pattern)

        elif func_name in self.SKLEARN_ENCODERS:
            pattern = DetectedPattern(
                pattern_type="encoder",
                name=func_name,
                source_file=str(file_path.relative_to(self.project_path)),
                source_line=node.lineno,
                generated_code=f"Feature(dtype='categorical', encoder={func_name}())",
            )
            self.detected_patterns.append(pattern)

    def _detect_sql_queries(self) -> None:
        """Detect SQL files for DataSource generation."""
        for sql_file in self.project_path.rglob("*.sql"):
            if ".venv" in str(sql_file):
                continue

            rel_path = sql_file.relative_to(self.project_path)
            name = sql_file.stem.replace("-", "_").replace(" ", "_")

            pattern = DetectedPattern(
                pattern_type="query",
                name=name,
                source_file=str(rel_path),
                source_line=1,
                generated_code=f'Query.from_file("{rel_path}")',
            )

            # Add TODO for source system configuration
            pattern.todos.append(TodoItem(
                message=f"Configure source system for query '{name}'",
                priority=TodoPriority.HIGH,
                file_path=str(rel_path),
                category="data_source",
            ))
            self.todos.append(pattern.todos[-1])

            self.detected_patterns.append(pattern)

    def _detect_data_files(self) -> None:
        """Detect CSV/Parquet files for file-based DataSources."""
        for ext in ["*.csv", "*.parquet", "*.pq"]:
            for data_file in self.project_path.rglob(ext):
                if ".venv" in str(data_file) or "__pycache__" in str(data_file):
                    continue

                rel_path = data_file.relative_to(self.project_path)
                name = data_file.stem.replace("-", "_").replace(" ", "_")

                pattern = DetectedPattern(
                    pattern_type="data_source",
                    name=name,
                    source_file=str(rel_path),
                    source_line=1,
                    generated_code=f'DataSource(name="{name}", source="file", path="{rel_path}")',
                )
                self.detected_patterns.append(pattern)

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Get the name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _generate_feature_set(self) -> str:
        """Generate FeatureSet class from detected patterns."""
        transformers = [p for p in self.detected_patterns if p.pattern_type == "transformer"]
        encoders = [p for p in self.detected_patterns if p.pattern_type == "encoder"]

        if not transformers and not encoders:
            self.todos.append(TodoItem(
                message="No sklearn transformers/encoders detected. Define features manually.",
                priority=TodoPriority.MEDIUM,
                category="features",
            ))
            return self._empty_feature_set()

        lines = [
            '"""Feature set generated by Geronimo import."""',
            "",
            "from geronimo.features import FeatureSet, Feature",
        ]

        # Collect unique imports
        sklearn_imports = set()
        for p in transformers + encoders:
            sklearn_imports.add(p.name)

        if sklearn_imports:
            imports_str = ", ".join(sorted(sklearn_imports))
            lines.append(f"from sklearn.preprocessing import {imports_str}")

        lines.extend([
            "",
            "",
            "class ProjectFeatures(FeatureSet):",
            '    """Auto-generated feature set.',
            "",
            "    TODO(MEDIUM): [features] Review and customize feature definitions.",
            '    """',
            "",
        ])

        # Add features
        for i, p in enumerate(transformers):
            lines.append(f"    # Detected at {p.source_file}:{p.source_line}")
            lines.append(f"    feature_{i} = {p.generated_code}")
            lines.append(f"    # TODO(MEDIUM): [features] Rename 'feature_{i}' to actual column name")
            lines.append("")

        for i, p in enumerate(encoders, start=len(transformers)):
            lines.append(f"    # Detected at {p.source_file}:{p.source_line}")
            lines.append(f"    feature_{i} = {p.generated_code}")
            lines.append(f"    # TODO(MEDIUM): [features] Rename 'feature_{i}' to actual column name")
            lines.append("")

        return "\n".join(lines)

    def _empty_feature_set(self) -> str:
        """Generate empty feature set template."""
        return '''"""Feature set for the project."""

from geronimo.features import FeatureSet, Feature
# TODO(HIGH): [features] Import transformers/encoders as needed
# from sklearn.preprocessing import StandardScaler, OneHotEncoder


class ProjectFeatures(FeatureSet):
    """Define your features here.

    TODO(HIGH): [features] Add feature definitions.

    Example:
        age = Feature(dtype='numeric', transformer=StandardScaler())
        category = Feature(dtype='categorical', encoder=OneHotEncoder())
    """

    pass
'''

    def _generate_data_sources(self) -> str:
        """Generate data sources from detected patterns."""
        queries = [p for p in self.detected_patterns if p.pattern_type == "query"]
        files = [p for p in self.detected_patterns if p.pattern_type == "data_source"]

        lines = [
            '"""Data sources generated by Geronimo import."""',
            "",
            "from geronimo.data import DataSource, Query",
            "",
        ]

        if not queries and not files:
            lines.append("# TODO(HIGH): [data] No data sources detected. Define manually.")
            lines.append("")
            lines.append('# training_data = DataSource(name="training", source="snowflake", query=Query.from_file("queries/train.sql"))')

            self.todos.append(TodoItem(
                message="No data sources detected. Define DataSource instances manually.",
                priority=TodoPriority.HIGH,
                category="data",
            ))
        else:
            for p in queries:
                lines.append(f"# Query detected: {p.source_file}")
                lines.append(f"{p.name}_query = {p.generated_code}")
                lines.append(f'# TODO(HIGH): [data] Configure source system for {p.name}')
                lines.append(f'{p.name} = DataSource(name="{p.name}", source="snowflake", query={p.name}_query)')
                lines.append(f'# TODO(MEDIUM): [data] Set correct source type: "snowflake", "postgres", or "sqlserver"')
                lines.append("")

            for p in files:
                lines.append(f"# Data file detected: {p.source_file}")
                lines.append(f"{p.name} = {p.generated_code}")
                lines.append("")

        return "\n".join(lines)

    def _generate_model(self) -> str:
        """Generate Model class template."""
        self.todos.append(TodoItem(
            message="Implement train() and predict() methods in Model class.",
            priority=TodoPriority.HIGH,
            category="model",
        ))

        return '''"""Model definition generated by Geronimo import."""

from geronimo.models import Model, HyperParams
from geronimo.artifacts import ArtifactStore

# TODO(HIGH): [model] Import your ML library
# from sklearn.ensemble import RandomForestClassifier


class ProjectModel(Model):
    """Main model class.

    TODO(HIGH): [model] Implement train() and predict() methods.
    """

    name = "project-model"
    version = "1.0.0"

    # TODO(MEDIUM): [model] Link to your FeatureSet
    # features = ProjectFeatures()

    def train(self, X, y, params: HyperParams) -> None:
        """Train the model.

        TODO(HIGH): [model] Implement training logic.

        Example:
            self.estimator = RandomForestClassifier(
                n_estimators=params.n_estimators,
                max_depth=params.max_depth,
            )
            self.estimator.fit(X, y)
        """
        raise NotImplementedError("Implement train() method")

    def predict(self, X):
        """Generate predictions.

        TODO(HIGH): [model] Implement prediction logic.

        Example:
            return self.estimator.predict_proba(X)
        """
        raise NotImplementedError("Implement predict() method")
'''

    def _generate_endpoint(self) -> str:
        """Generate Endpoint class template."""
        return '''"""Endpoint definition generated by Geronimo import."""

from geronimo.serving import Endpoint

# TODO(MEDIUM): [endpoint] Import your model class
# from .model import ProjectModel


class PredictEndpoint(Endpoint):
    """Prediction endpoint.

    TODO(HIGH): [endpoint] Configure model class.
    """

    # TODO(HIGH): [endpoint] Set your model class
    # model_class = ProjectModel

    def preprocess(self, request: dict):
        """Preprocess incoming request.

        TODO(HIGH): [endpoint] Implement preprocessing.

        Example:
            import pandas as pd
            df = pd.DataFrame([request["data"]])
            return self.model.features.transform(df)
        """
        raise NotImplementedError("Implement preprocess() method")

    def postprocess(self, prediction):
        """Postprocess model output.

        TODO(HIGH): [endpoint] Implement postprocessing.

        Example:
            return {
                "score": float(prediction[0]),
                "class": "positive" if prediction[0] > 0.5 else "negative",
            }
        """
        raise NotImplementedError("Implement postprocess() method")
'''

    def generate_summary(self) -> str:
        """Generate summary of detected patterns and TODOs."""
        lines = [
            "# Import Summary",
            "",
            "## Detected Patterns",
            "",
        ]

        by_type = {}
        for p in self.detected_patterns:
            by_type.setdefault(p.pattern_type, []).append(p)

        for ptype, patterns in by_type.items():
            lines.append(f"### {ptype.title()}s ({len(patterns)})")
            for p in patterns[:5]:
                lines.append(f"- `{p.name}` at {p.source_file}:{p.source_line}")
            if len(patterns) > 5:
                lines.append(f"- ... and {len(patterns) - 5} more")
            lines.append("")

        lines.extend([
            "## TODO Items",
            "",
        ])

        high = [t for t in self.todos if t.priority == TodoPriority.HIGH]
        medium = [t for t in self.todos if t.priority == TodoPriority.MEDIUM]
        low = [t for t in self.todos if t.priority == TodoPriority.LOW]

        if high:
            lines.append(f"### HIGH Priority ({len(high)})")
            for t in high:
                lines.append(f"- [{t.category}] {t.message}")
            lines.append("")

        if medium:
            lines.append(f"### MEDIUM Priority ({len(medium)})")
            for t in medium:
                lines.append(f"- [{t.category}] {t.message}")
            lines.append("")

        if low:
            lines.append(f"### LOW Priority ({len(low)})")
            for t in low:
                lines.append(f"- [{t.category}] {t.message}")
            lines.append("")

        return "\n".join(lines)
