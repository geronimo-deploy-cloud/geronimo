# API Documentation

This directory contains auto-generated API documentation for the Geronimo library.

## Generating Documentation

### Using the CLI

```bash
# Generate static HTML documentation
geronimo docs generate

# Generate to custom output directory
geronimo docs generate --output ./build/api-docs

# Start a live-reloading development server
geronimo docs serve

# Serve on a custom port
geronimo docs serve --port 3000
```

### Using the Script

```bash
# From the project root
python docs/generate_docs.py

# Custom output directory
python docs/generate_docs.py --output ./build/api-docs

# Live preview server
python docs/generate_docs.py --serve
```

## Output Structure

The generated documentation follows this structure:

```
docs/api/
├── index.html              # Main index page
├── geronimo/
│   ├── artifacts.html      # Artifact store docs
│   ├── batch.html          # BatchPipeline docs
│   ├── cli.html            # CLI command docs
│   ├── config.html         # Configuration docs
│   ├── data_sources.html   # Data sources and connections docs
│   ├── deploy.html         # Deployment target docs
│   ├── deploy_cloud.html   # Cloud SDK docs
│   ├── features.html       # Feature engineering docs
│   ├── generators.html     # Code generator docs
│   ├── models.html         # Model base class docs
│   ├── serving.html        # Endpoint serving docs
│   └── validation.html     # Validation engine docs
└── search.js               # Client-side search
```

## Website Integration

### For the geronimo-oss-website (Next.js)

1. **Copy generated files** to the website's public directory:

   ```bash
   # From the geronimo repo root
   geronimo docs generate --output ../geronimo-oss-website/public/api-docs
   ```

2. **Add iframe or static hosting** in Next.js:

   Option A: Serve as static files at `/api-docs/`
   ```
   # Files will be available at https://geronimo.dev/api-docs/
   ```

   Option B: Create a dedicated API docs page with iframe
   ```jsx
   // src/app/docs/api/page.tsx
   export default function APIDocsPage() {
     return (
       <iframe 
         src="/api-docs/index.html" 
         className="w-full h-screen border-0"
       />
     );
   }
   ```

3. **Add navigation link** in the docs sidebar to `/docs/api` or `/api-docs/`

### CI/CD Integration

Add a GitHub Actions workflow to regenerate docs on release:

```yaml
# .github/workflows/docs.yml
name: Generate API Docs

on:
  release:
    types: [published]
  push:
    branches: [main]
    paths:
      - 'src/geronimo/**'

jobs:
  generate-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install uv
          uv sync --group dev
      
      - name: Generate documentation
        run: uv run python scripts/generate_docs.py --output ./docs/api
      
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: api-docs
          path: docs/api/
```

## Covered Modules

The following public modules are documented:

| Module | Description |
|--------|-------------|
| `geronimo.artifacts` | MLflow-backed artifact storage |
| `geronimo.batch` | BatchPipeline for scheduled jobs |
| `geronimo.cli` | Command-line interface |
| `geronimo.config` | YAML configuration loading |
| `geronimo.data_sources` | Data sources and connections |
| `geronimo.deploy` | Deployment targets and protocols |
| `geronimo.deploy_cloud` | Geronimo Cloud SDK |
| `geronimo.features` | Feature engineering |
| `geronimo.generators` | Terraform, Docker, pipeline generators |
| `geronimo.models` | Model base class |
| `geronimo.serving` | Endpoint serving |
| `geronimo.validation` | Configuration validation |

## Requirements

- Python 3.11+
- pdoc >= 15.0.0 (installed as dev dependency)
