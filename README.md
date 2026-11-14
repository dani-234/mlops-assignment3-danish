# Assignment 03 - Production MLOps AI Agent

This repository implements an offline-capable FastAPI MLOps agent with DVC provenance,
MLflow run intelligence, Airflow 3 API helpers, Kubernetes diagnostics, deterministic
RandomForest training, confirmation-protected mutations, and failure fixtures.

## Setup

Use Python 3.11, create a virtual environment, then run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
.venv\Scripts\python training\generate_dataset.py
.venv\Scripts\python -m training.train
```

The repository includes a local no-SCM DVC configuration, generated `.dvc` metadata,
`dvc.lock`, and `params.yaml`. Validate and reproduce the pipeline with
`python -m dvc status` and `python -m dvc repro`. A remote is optional for local tests.
Airflow 3 is isolated from application dependencies. Install it with the official
Python 3.11 constraints command in `requirements-airflow.txt`, then copy
`dags/training_pipeline.py` into the Airflow DAG folder. MLflow and Airflow are
optional for offline tests. The application uses `mlflow-skinny` for tracking-client
operations; install the optional `mlflow-server` extra only when running a local
MLflow server from Python rather than the provided Compose server image.

Training writes `artifacts/model.joblib`, `artifacts/preprocessor.joblib`,
`artifacts/processed_data.joblib`, and `artifacts/model_metadata.json`. The scaler is fit
only on the training split and the serving API applies that same persisted transformer.
Metadata contains metrics, the DVC checksum, the Git commit when available, and the
quality-gate decision. The Airflow evaluation stage logs metrics, artifacts, and provenance
tags to MLflow when its tracking service is configured. DVC reproduction can be explained
offline or explicitly confirmed for execution in a temporary Git worktree; it never
changes the working copy.

## Run and verify

```powershell
.venv\Scripts\python -m uvicorn api.main:app --port 8000
.venv\Scripts\python -m agent.main "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
.venv\Scripts\python -m agent.main --failure quality-gate
.venv\Scripts\python -m agent.main --failure kubernetes
.venv\Scripts\python -m pytest -v
.venv\Scripts\python -m ruff check .
```

`/health/live` only checks process liveness; `/health/ready` requires the configured model
artifact. `/predict` requires four numeric features. `/agent/query` uses the deterministic
planner when no LLM provider is configured. Set `AGENT_LLM_PROVIDER=openai` and
`OPENAI_API_KEY` to enable the restricted OpenAI tool-calling path; provider failures
fall back to the deterministic planner and are returned as uncertainties. Live MLflow,
Airflow, and Kubernetes results require their respective environment variables and
services. Kubernetes deployment mutations require an approved namespace and a single-use
confirmation token, and return a bounded rollout verification result.

## Deployment

`docker compose up --build` starts the API and MLflow. The Dockerfile trains the
deterministic model during image build, so the serving artifact is present without
training on API startup. Compose is not Kubernetes; for a
local cluster build/load `assignment-03-mlops-agent:local`, then apply
`kubernetes/namespace.yaml`, `configmap.yaml`, `deployment.yaml`, and `service.yaml`.
The model artifact must be supplied through the documented `artifacts` volume or image
build contract. Never place credentials in ConfigMaps.

## Evidence runbook

Run these commands from the project root and save the terminal output as evidence.
Label output **simulated** when the referenced service is unavailable and **live** only
when the service returned the result.

### Checkpoint 1 - local deterministic evidence

```powershell
python -m agent.main "Show dataset provenance"
python -m agent.main "Find the best recent model and explain whether retraining is necessary"
python -m agent.main --failure quality-gate
python -m agent.main --failure kubernetes
```

Capture one terminal screenshot showing each command and its JSON result.

### Checkpoint 2 - MLflow

```powershell
docker compose up --build -d
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python -m training.train
python -m agent.main "Find the best recent model"
```

Capture the MLflow run ID, metrics, artifact paths, and provenance tags. Stop Compose
after the evidence is saved with `docker compose down`.

### Checkpoint 3 - Airflow

Install Airflow separately using `requirements-airflow.txt`, start the Airflow 3
standalone/API service, place `dags/training_pipeline.py` in its DAG folder, configure
`AIRFLOW_API_URL` and authentication, then run:

```powershell
python -m agent.main "Check the training pipeline status"
```

Capture the DAG ID, run ID, task states, and quality-gate result. Label this evidence
live only if it came from the Airflow API.

### Checkpoint 4 - Kubernetes

Build/load the image into Kind or Minikube, apply the manifests, and run:

```powershell
kubectl -n mlops-prod get pods,deployments,services
python -m agent.main "Check deployment status and diagnose unhealthy pods"
```

Capture Pod readiness, Deployment replicas, Service details, image/model metadata, and
any diagnostic evidence. Do not execute scale/restart commands for the assignment
evidence unless you intentionally capture the confirmation rejection and authorized
mutation separately.

### Checkpoint 5 - multi-tool reasoning

```powershell
python -m agent.main "Find the best recent model, identify the dataset version used to train it, check deployment status, inspect pipeline status, and explain whether retraining is necessary"
```

Capture the complete tool trace and label unavailable services explicitly as uncertainty.

