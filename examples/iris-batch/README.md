# Iris Batch Pipeline Example

A proper batch job example using Metaflow for orchestrated ML scoring.

## Overview

This example demonstrates how to build a **scheduled batch scoring pipeline** that:
- Runs on a fixed schedule (daily at 6 AM)
- Loads data from a source
- Applies feature transformations
- Generates predictions in batch
- Saves results to storage

## Project Structure

```
iris-batch/
├── flows/
│   └── scoring_flow.py      # Metaflow DAG definition
├── src/iris_batch/
│   ├── pipeline.py          # BatchPipeline implementation
│   ├── features.py          # Feature definitions
│   └── model.py             # Model wrapper
├── models/                   # Trained model artifacts
├── data/                     # Input data
├── output/                   # Batch results
└── geronimo.yaml            # Configuration
```

## Running the Pipeline

### Local Execution
```bash
python flows/scoring_flow.py run
```

### Scheduled Execution (AWS Step Functions)
```bash
python flows/scoring_flow.py step-functions create
```

### Airflow Deployment
```bash
geronimo generate batch --target airflow
```

## Key Concepts

- **BatchPipeline**: Geronimo's base class for batch jobs
- **Schedule**: Cron-based scheduling (daily, weekly, custom)
- **Trigger**: Event-based execution (S3 upload, SNS message)
