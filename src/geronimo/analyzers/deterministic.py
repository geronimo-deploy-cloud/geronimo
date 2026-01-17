"""Deterministic analyzer for Geronimo.

Uses AST parsing and pattern matching to analyze project structure.
This is the default analyzer that works without any AI integration.
"""

import ast
import re
from pathlib import Path

from geronimo.analyzers.base import (
    AnalysisResult,
    Analyzer,
    EndpointInfo,
    ModelArtifactInfo,
    PreprocessingStep,
    ServiceInfo,
)
from geronimo.config.schema import MLFramework
from geronimo.scanners.project import ProjectScan


class DeterministicAnalyzer(Analyzer):
    """Rule-based project analyzer using AST parsing.

    Analyzes projects by:
    1. Detecting ML framework from dependencies
    2. Parsing Python files to find FastAPI endpoints
    3. Tracing service imports to build preprocessing chain
    4. Identifying model loading patterns
    """

    # Dependency patterns for framework detection
    FRAMEWORK_PATTERNS: dict[MLFramework, list[str]] = {
        MLFramework.SKLEARN: ["scikit-learn", "sklearn"],
        MLFramework.PYTORCH: ["torch", "pytorch"],
        MLFramework.TENSORFLOW: ["tensorflow", "keras"],
        MLFramework.XGBOOST: ["xgboost"],
    }

    # FastAPI decorator patterns
    ROUTE_DECORATORS = {"get", "post", "put", "delete", "patch", "options", "head"}

    @property
    def analyzer_type(self) -> str:
        return "deterministic"

    def analyze(self, project: ProjectScan) -> AnalysisResult:
        """Analyze the project using deterministic rules."""
        # Detect framework from dependencies
        framework = self._detect_framework(project.dependencies)

        # Analyze endpoints
        endpoints = self._find_endpoints(project)

        # Analyze services
        services = self._analyze_services(project)

        # Build preprocessing chain
        preprocessing = self._build_preprocessing_chain(services, endpoints)

        # Analyze model artifacts
        artifacts = self._analyze_model_artifacts(project, framework)

        # Calculate confidence and identify items needing review
        confidence, needs_review, warnings = self._assess_confidence(
            framework, endpoints, services, artifacts
        )

        return AnalysisResult(
            detected_framework=framework,
            python_version=project.python_version,
            dependencies=project.dependencies,
            endpoints=endpoints,
            services=services,
            preprocessing_chain=preprocessing,
            model_artifacts=artifacts,
            confidence=confidence,
            needs_review=needs_review,
            warnings=warnings,
            analyzer_type=self.analyzer_type,
        )

    def _detect_framework(self, dependencies: list[str]) -> MLFramework:
        """Detect ML framework from dependencies."""
        dep_lower = [d.lower() for d in dependencies]
        dep_str = " ".join(dep_lower)

        for framework, patterns in self.FRAMEWORK_PATTERNS.items():
            for pattern in patterns:
                if pattern in dep_str:
                    return framework

        return MLFramework.CUSTOM

    def _find_endpoints(self, project: ProjectScan) -> list[EndpointInfo]:
        """Find FastAPI endpoints by parsing route files."""
        endpoints = []

        # Look in routes directory first, then api directory
        search_dirs = []
        if project.routes_dir:
            search_dirs.append(project.routes_dir)
        if project.api_dir:
            search_dirs.append(project.api_dir)
        if project.src_package:
            search_dirs.append(project.src_package)

        for search_dir in search_dirs:
            if not search_dir or not search_dir.exists():
                continue

            for py_file in search_dir.rglob("*.py"):
                try:
                    file_endpoints = self._parse_endpoints_from_file(py_file)
                    endpoints.extend(file_endpoints)
                except Exception:
                    # Skip files that can't be parsed
                    continue

        return endpoints

    def _parse_endpoints_from_file(self, file_path: Path) -> list[EndpointInfo]:
        """Parse a Python file to find FastAPI route decorators."""
        endpoints = []

        try:
            source = file_path.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            return []

        # Find the router variable name (e.g., router = APIRouter())
        router_names = {"app", "router"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Call):
                            call_name = self._get_call_name(node.value)
                            if call_name in ("APIRouter", "FastAPI"):
                                router_names.add(target.id)

        # Find decorated functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                for decorator in node.decorator_list:
                    endpoint_info = self._parse_route_decorator(
                        decorator, node, file_path, router_names
                    )
                    if endpoint_info:
                        endpoints.append(endpoint_info)

        return endpoints

    def _parse_route_decorator(
        self,
        decorator: ast.expr,
        func: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: Path,
        router_names: set[str],
    ) -> EndpointInfo | None:
        """Parse a route decorator to extract endpoint info."""
        # Handle @router.get("/path") style
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute):
                if (
                    isinstance(decorator.func.value, ast.Name)
                    and decorator.func.value.id in router_names
                    and decorator.func.attr in self.ROUTE_DECORATORS
                ):
                    # Extract path from first argument
                    path = "/"
                    if decorator.args:
                        if isinstance(decorator.args[0], ast.Constant):
                            path = decorator.args[0].value

                    # Extract schemas from decorator kwargs
                    request_schema = None
                    response_schema = None
                    for kw in decorator.keywords:
                        if kw.arg == "response_model":
                            response_schema = self._get_annotation_name(kw.value)

                    # Try to get request schema from function parameters
                    for arg in func.args.args:
                        if arg.annotation and arg.arg not in ("self", "request"):
                            request_schema = self._get_annotation_name(arg.annotation)
                            break

                    return EndpointInfo(
                        path=path,
                        method=decorator.func.attr.upper(),
                        handler_function=func.name,
                        file_path=file_path,
                        request_schema=request_schema,
                        response_schema=response_schema,
                        description=ast.get_docstring(func),
                    )

        return None

    def _get_call_name(self, node: ast.Call) -> str:
        """Get the name of a function being called."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _get_annotation_name(self, node: ast.expr) -> str | None:
        """Get the name from a type annotation."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript):
            return self._get_annotation_name(node.value)
        return None

    def _analyze_services(self, project: ProjectScan) -> list[ServiceInfo]:
        """Analyze service modules."""
        services = []

        if not project.services_dir or not project.services_dir.exists():
            return services

        for py_file in project.services_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                service = self._parse_service_file(py_file, project)
                if service.functions or service.classes:
                    services.append(service)
            except Exception:
                continue

        return services

    def _parse_service_file(self, file_path: Path, project: ProjectScan) -> ServiceInfo:
        """Parse a service file to extract functions and classes."""
        source = file_path.read_text()
        tree = ast.parse(source)

        functions = []
        classes = []
        dependencies = []

        for node in ast.walk(tree):
            # Find public functions
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if not node.name.startswith("_"):
                    functions.append(node.name)

            # Find classes
            if isinstance(node, ast.ClassDef):
                if not node.name.startswith("_"):
                    classes.append(node.name)

            # Find imports from project
            if isinstance(node, ast.ImportFrom):
                if node.module and project.project_name.replace("-", "_") in node.module:
                    dependencies.append(node.module)

        return ServiceInfo(
            name=file_path.stem,
            file_path=file_path,
            functions=functions,
            classes=classes,
            dependencies=dependencies,
        )

    def _build_preprocessing_chain(
        self,
        services: list[ServiceInfo],
        endpoints: list[EndpointInfo],
    ) -> list[PreprocessingStep]:
        """Build preprocessing chain from services.

        Uses heuristics to determine the order of preprocessing steps.
        """
        steps = []
        order = 0

        # Common preprocessing-related terms (in likely order)
        preprocessing_patterns = [
            ("load", "data loading"),
            ("fetch", "data fetching"),
            ("clean", "data cleaning"),
            ("preprocess", "preprocessing"),
            ("transform", "transformation"),
            ("feature", "feature engineering"),
            ("encode", "encoding"),
            ("scale", "scaling"),
            ("normalize", "normalization"),
            ("predict", "prediction"),
            ("infer", "inference"),
        ]

        for service in services:
            # Check if this service is preprocessing-related
            service_name_lower = service.name.lower()

            for pattern, description in preprocessing_patterns:
                if pattern in service_name_lower:
                    # Add functions that match the pattern
                    for func in service.functions:
                        steps.append(
                            PreprocessingStep(
                                name=f"{service.name}.{func}",
                                source_file=service.file_path,
                                source_function=func,
                                order=order,
                            )
                        )
                    order += 1
                    break

        # Sort by order
        steps.sort(key=lambda s: s.order)
        return steps

    def _analyze_model_artifacts(
        self,
        project: ProjectScan,
        framework: MLFramework,
    ) -> list[ModelArtifactInfo]:
        """Analyze model artifact files."""
        artifacts = []

        for path in project.model_artifacts:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0

            # Detect framework from extension
            artifact_framework = framework
            ext = path.suffix.lower()
            if ext in (".pt", ".pth"):
                artifact_framework = MLFramework.PYTORCH
            elif ext in (".h5", ".keras"):
                artifact_framework = MLFramework.TENSORFLOW

            # Determine if this is the primary model
            is_primary = len(project.model_artifacts) == 1 or "model" in path.stem.lower()

            artifacts.append(
                ModelArtifactInfo(
                    path=path,
                    framework=artifact_framework,
                    size_bytes=size,
                    is_primary=is_primary,
                )
            )

        return artifacts

    def _assess_confidence(
        self,
        framework: MLFramework,
        endpoints: list[EndpointInfo],
        services: list[ServiceInfo],
        artifacts: list[ModelArtifactInfo],
    ) -> tuple[float, list[str], list[str]]:
        """Assess confidence in the analysis and identify items needing review."""
        confidence = 1.0
        needs_review = []
        warnings = []

        # Framework detection
        if framework == MLFramework.CUSTOM:
            confidence -= 0.2
            needs_review.append("Could not detect ML framework - please specify manually")

        # Endpoint detection
        if not endpoints:
            confidence -= 0.1
            warnings.append("No API endpoints detected")

        # Check for prediction endpoint
        has_predict = any("predict" in e.path.lower() or "predict" in e.handler_function.lower() for e in endpoints)
        if endpoints and not has_predict:
            warnings.append("No prediction endpoint found - is this an ML serving API?")

        # Services
        if not services:
            confidence -= 0.1
            warnings.append("No services directory found")

        # Model artifacts
        if not artifacts:
            confidence -= 0.2
            needs_review.append("No model artifacts found - add your model to models/")
        elif len(artifacts) > 1:
            warnings.append(f"Multiple model artifacts found ({len(artifacts)}) - using largest as primary")

        return max(0.0, confidence), needs_review, warnings
