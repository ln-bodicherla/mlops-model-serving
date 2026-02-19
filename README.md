# MLOps Model Serving Pipeline

An end-to-end ML model lifecycle management platform featuring MLflow experiment tracking, FastAPI model serving, Docker containerization, automated A/B testing, and data drift monitoring. Provides a production-grade infrastructure for deploying, monitoring, and iterating on machine learning models.

## Overview

This project implements the full MLOps lifecycle:
- **Experiment Tracking**: Log parameters, metrics, and artifacts with MLflow
- **Model Registry**: Version control for models with stage transitions
- **Model Serving**: Low-latency FastAPI server with request validation
- **A/B Testing**: Statistical testing framework for model comparison
- **Drift Detection**: Real-time monitoring for data and prediction drift
- **Containerization**: Docker-based deployment with service orchestration

## Architecture

```
Training Pipeline
       |
       v
MLflow Tracking Server ──> Model Registry
       |                        |
       v                        v
Experiment Comparison     Model Versioning
                               |
                          ┌────┴────┐
                          v         v
                    Champion    Challenger
                          |         |
                          v         v
                     A/B Test Router
                          |
                          v
                   FastAPI Model Server
                          |
                     ┌────┴────┐
                     v         v
              Drift Monitor  Prometheus Metrics
                     |
                     v
              Alerting System
```

## Project Structure

```
mlops-model-serving/
├── configs/
│   └── config.yaml
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/
│   ├── experiment_tracker.py   # MLflow experiment management
│   ├── model_server.py         # FastAPI serving application
│   ├── drift_monitor.py        # Data drift detection
│   └── ab_testing.py           # A/B testing framework
├── tests/
│   └── test_server.py          # API endpoint tests
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Start the Infrastructure

```bash
docker-compose -f docker/docker-compose.yml up -d
```

### Track an Experiment

```python
from src.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(tracking_uri="http://localhost:5000")
tracker.setup_mlflow(experiment_name="my_experiment")

with tracker.log_training_run(run_name="baseline_v1") as run:
    # Train your model
    tracker.log_params({"learning_rate": 0.01, "n_estimators": 100})
    tracker.log_metrics({"accuracy": 0.95, "f1_score": 0.93})
    tracker.log_model(model, "sklearn_model")
```

### Serve the Model

```bash
uvicorn src.model_server:app --host 0.0.0.0 --port 8000
```

### Monitor for Drift

```python
from src.drift_monitor import DriftMonitor

monitor = DriftMonitor()
monitor.setup_reference_data(training_data)
drift_report = monitor.detect_data_drift(new_data)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| POST | `/predict/ab` | A/B test routed prediction |
| GET | `/model/info` | Current model metadata |
| GET | `/model/health` | Health check with validation |
| GET | `/metrics` | Prometheus metrics |

## Configuration

See `configs/config.yaml` for all configurable parameters including MLflow settings, server configuration, drift thresholds, and A/B test parameters.

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT License
