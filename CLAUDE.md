# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Geronimo is a declarative MLOps framework (think "dbt for AI"). The library publishes:

1. **An SDK** consumed inside scaffolded user projects (`DataSource`, `FeatureSet`, `Model`, `MonitoringConfig`, `Endpoint`, `BatchPipeline`).
2. **A `geronimo` CLI** that scaffolds those projects, validates them, generates deployment artifacts (Terraform/Docker/CI), and runs Pulumi-based deploys (AWS/GCP/Azure/GDC).
3. **An artifact store abstraction** with pluggable backends (local / S3 / MLflow / GDC), driven by `~/.geronimo/config.yaml`.

Two project shapes are generated: `realtime` (FastAPI + fastmcp MCP server) and `batch` (Metaflow + Evidently drift).

## Common commands

This project is uv-managed. Run everything through `uv run`.

```bash
uv sync --all-extras --dev          # set up environment (matches CI)
uv run pytest tests/ -v             # run all tests
uv run pytest tests/test_cli.py     # run a single test file
uv run pytest tests/test_cli.py::test_name  # run one test
uv run pytest tests/ --cov=src/geronimo --cov-report=term-missing  # coverage (matches CI)
uv run mypy src/geronimo            # type-check
uv run python docs/generate_docs.py            # build HTML API docs into docs/api
uv run python docs/generate_docs.py --serve    # live preview on localhost:8080
uv run geronimo --version           # exercise the CLI in dev
```

Tests run end-to-end on every PR via `.github/workflows/tests.yml` — there is no separate lint step in CI, but mypy is configured in `pyproject.toml`.

## Architecture

### Package layout (`src/geronimo/`)

The package is sliced by **capability**, not by layer. Each subpackage is independently importable and most are listed in `geronimo/constants.py::MODULES` (the canonical list of public modules — keep it in sync when adding/removing top-level packages, because `docs/generate_docs.py` reads it).

- `cli/` — Typer app. `main.py` is the root; it mounts subcommand groups (`generate`, `keys`, `auth`, `config`, `deploy`) from sibling modules via `app.add_typer(...)`. `docs_app` is intentionally **not** registered.
- `config/` — Pydantic schema for `geronimo.yaml` (`schema.py`), file loader, plus `user_config.py` for the global `~/.geronimo/config.yaml`.
- `data_sources/`, `features/`, `models/`, `batch/`, `serving/` — the user-facing SDK surface. These are imported *by generated user projects*, so changing public signatures here is a breaking change for downstream scaffolds.
- `serving/auth/` — API-key middleware + keys store used by FastAPI scaffolds.
- `artifacts/` — Versioned artifact store. `protocol.py` defines the `ArtifactBackend` Protocol; `local_backend.py`, `s3_backend.py`, `mlflow_backend.py`, `gdc_backend.py` implement it; `store.py` is the user-facing `ArtifactStore` that picks a backend from config. Adding a backend means implementing the Protocol and wiring it into `store.py`.
- `generators/` — Jinja2-driven code generation. `BaseGenerator` (in `base.py`) wires the `PackageLoader("geronimo.generators", "templates")`. Concrete generators (`project.py`, `docker.py`, `terraform.py`, `pipeline.py`) render trees under `generators/templates/` (`project/`, `sdk/`, `docker/`, `terraform/`, `pipeline/`, `monitoring/`). When you add a new template file, it ships only if it lives under `generators/templates/` (hatchling packages the whole `src/geronimo` tree).
- `deploy/` — Pulumi-based deployment. `targets.py::deploy()` does **runtime detection** of the `pulumi` package and raises `PulumiNotInstalledError` if it's missing (Pulumi is an optional extra). Providers under `deploy/providers/{aws,gcp,azure}.py` are imported lazily inside `deploy()` so the base install stays light.
- `deploy_cloud/` — HTTP client for the managed "Geronimo Deploy Cloud" (GDC) backend. This is distinct from `deploy/` (Pulumi). The `cloud` backend was renamed to `deploy_cloud` in v0.4.0 — do not reintroduce `cloud` as a name.
- `deploy_testing_fixtures/` — Test doubles for the deploy clients, re-exported by `tests/conftest.py`.
- `validation/` — `geronimo validate` rules engine (`engine.py` + `rules.py`).

### How a user project comes to life

1. `geronimo init` → `generators/project.py::ProjectGenerator` renders `templates/project/` + `templates/sdk/` into a new directory, writes `geronimo.yaml` via `config.loader.save_config`, and selects deps from `CORE_DEPS` / `REALTIME_DEPS` / `BATCH_DEPS` based on `--template`.
2. The user fills in `sdk/{data_sources,features,model,monitoring_config}.py` (plus `endpoint.py` for realtime or `pipeline.py` for batch).
3. `geronimo validate` walks the project and runs `validation/rules.py` against the parsed `geronimo.yaml`.
4. `geronimo generate {terraform,dockerfile,pipeline,all}` produces deployment artifacts from the same `geronimo.yaml`.
5. `geronimo deploy up --target {aws,gcp,azure,gdc}` either invokes Pulumi (first three) or POSTs to GDC via `deploy_cloud/client.py`.

`geronimo.yaml` is the single source of truth for both generate and deploy; its schema lives in `config/schema.py`.

### Optional-dependency boundaries

Heavy / niche deps are gated behind extras in `pyproject.toml`: `[mlflow]`, `[databases]` (snowflake/postgres/odbc), `[pulumi]`, `[jwt]`, `[testing]`, `[all]`. Code paths that need them must import lazily inside the function that uses them and surface an actionable error if the import fails (see `deploy/targets.py::_check_pulumi_available` for the pattern). Do not add these deps to the top-level `dependencies` list.

### Tests

`tests/conftest.py` provides the shared fixtures: `temp_dir`, `sample_df` / `iris_df`, `geronimo_config` (writes a minimal `geronimo.yaml`), `temp_artifact_store` (real local backend in a tmp dir), and `mock_*` factories that delegate to `geronimo.deploy_testing_fixtures` so production code and tests share the same mock shapes. Prefer these fixtures over ad-hoc mocks.

## Conventions worth knowing

- Public modules in `constants.py::MODULES` are what `pdoc` documents. Adding a top-level package without listing it there means it won't appear in the generated API docs.
- Generated project scaffolds live under `examples/iris-realtime` and `examples/iris-batch` and double as integration references — if you change SDK signatures, update both.
- The CLI uses `rich` for output; prefer `console.print(...)` and `typer.Exit(code=1)` over `print` + `sys.exit` to stay consistent with the rest of `cli/main.py`.
- `__version__` lives in `src/geronimo/__init__.py` — bump it there for releases.

# Development Philosophy

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.