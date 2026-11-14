# MASTER PROMPT — Assignment 03: Production-Ready MLOps AI Agent

## ROLE AND MISSION

You are a Staff MLOps Engineer, Production AI Systems Architect, Python Backend Engineer, and Test Automation Engineer.

Your task is to design, implement, test, document, and validate a complete, reproducible MLOps AI Agent for:

**Assignment No. 3: Building a Production-Ready MLOps AI Agent**  
Course: MLOps & Production AI Engineering — Fall 2026  
Programming language: Python 3.11  
Total marks: 100

The system must provide a conversational interface through which a user can inspect dataset provenance, compare machine-learning experiments, operate training pipelines, diagnose Kubernetes deployments, and obtain evidence-based recommendations.

Required technologies:

- DVC for dataset versioning and provenance.
- MLflow for experiment tracking, evaluation metrics, model artifacts, and model registry information.
- Apache Airflow 3 for training workflow orchestration.
- Kubernetes for deploying and diagnosing the inference service.
- FastAPI for both model inference and agent interaction.
- Docker for application containerization.
- GitHub Actions for continuous integration.
- Pydantic v2 for data validation.
- pytest and Ruff for automated verification.

The objective is to deliver a functional repository, not merely a design, code skeleton, or collection of functions that appear complete.

## 1. NON-NEGOTIABLE IMPLEMENTATION RULES

1. Inspect the existing workspace before changing files. If the workspace is already the repository root, work there rather than creating a redundant nested project directory. Preserve useful existing work unless it conflicts with the requirements.

2. Implement the entire repository. Do not produce placeholders, TODO comments, empty functions, mock-only production implementations, incomplete files, or pseudocode in place of required functionality.

3. Do not claim the application is bug-free or production-ready merely because code was generated. Verify what can actually be verified and report remaining risks honestly.

4. Use supported Python APIs and official service interfaces. Do not scrape web interfaces, fabricate API responses, or invent operational facts.

5. Before implementing integrations, check the official documentation for the exact pinned versions of Airflow, MLflow, DVC, Kubernetes Python client, FastAPI, and their relevant APIs. Resolve incompatible versions before writing dependent code. Record the selected versions and installation commands.

6. Use Python 3.11 throughout the application, tests, and CI unless an official dependency constraint for a separately isolated service requires otherwise.

7. Keep service integrations modular. Business logic, external service clients, request schemas, API endpoints, orchestration, and tests must have clearly defined responsibilities.

8. Keep all secrets out of source code, generated reports, screenshots, Git history, and logs. Load credentials from environment variables or approved secret-management mechanisms.

9. Never allow a natural-language request or an LLM-generated tool call to bypass input validation, permission checks, command allow-lists, or operational safety controls.

10. If a service is unavailable, return a truthful and structured error. Never replace missing operational results with guessed metrics, synthetic Pod names, fabricated commit hashes, invented DAG runs, or assumed deployment statuses.

11. Execute the required lint and test commands in the actual workspace. Fix the failures that can be fixed, rerun the checks, and provide the actual results.

12. Do not stop after creating files. Continue through dependency setup, static inspection, testing, debugging, documentation, and the final verification report.

13. If a live service, credential, cluster, network connection, or external dependency is unavailable, continue with all feasible offline implementation and testing. Report exactly what was not verified. Do not ask unnecessary questions when reasonable, documented defaults can be selected.

## 2. REQUIRED REPOSITORY STRUCTURE

Create and maintain the following structure. Additional files are permitted when they are necessary for a coherent, reproducible implementation.

```text
assignment-03-mlops-agent/
├── agent/
│   ├── __init__.py
│   ├── main.py
│   ├── orchestrator.py
│   ├── schemas.py
│   ├── logger.py
│   ├── failure_simulator.py
│   └── tools/
│       ├── __init__.py
│       ├── dvc_tools.py
│       ├── mlflow_tools.py
│       ├── airflow_tools.py
│       └── kubernetes_tools.py
├── api/
│   ├── __init__.py
│   └── main.py
├── dags/
│   └── training_pipeline.py
├── data/
│   ├── raw_dataset.csv
│   └── raw_dataset.csv.dvc
├── training/
│   ├── __init__.py
│   ├── generate_dataset.py
│   └── train.py
├── kubernetes/
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
├── tests/
│   ├── __init__.py
│   └── test_agent_and_tools.py
├── screenshots/
│   └── README.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── pyproject.toml
├── requirements-airflow.txt
├── README.md
└── architecture.md
```

Generate any additional configuration, locking, scripts, evidence directories, or supporting files that are necessary. Do not omit a required file because it is absent from this list.

Do not commit generated model binaries, local DVC caches, runtime databases, credentials, Kubernetes credentials, or other environment-specific files unless there is a specific, justified need to include them.

## 3. DEPENDENCIES AND REPRODUCIBLE ENVIRONMENT

### 3.1 Python application

Configure `pyproject.toml` with project metadata, Python version requirements, dependencies, development tooling, pytest configuration, and Ruff settings.

Include compatible versions of the direct application dependencies required by the implementation, including:

- fastapi
- uvicorn
- pydantic v2
- mlflow
- dvc
- httpx
- kubernetes
- scikit-learn
- pandas
- numpy
- openai
- google-genai, if using the Google Gemini API
- pytest
- ruff
- joblib, if used for model artifact serialization

Pin dependencies or define explicit compatible constraints, then generate and maintain a real lock file, such as `uv.lock`, where appropriate. Do not invent lock-file contents.

Use the official installation constraints for the selected Airflow version. Avoid blindly installing all application and Airflow dependencies into one environment if that causes dependency conflicts. Document and implement a reproducible separation, such as a dedicated Airflow container or an isolated Airflow environment.

Use Python 3.11 and a compatible Airflow 3 release. Pin the selected Airflow version and use the official, version-specific constraints for that Python version. Verify the selected version and installation method using the official Airflow documentation.

Airflow API interactions must follow the documented interface for the installed Airflow version. The implementation must not silently assume that every API version is interchangeable.

### 3.2 Local development environment

Provide `.env.example` with safe, non-secret configuration values and explanations.

Include a useful `docker-compose.yml` for the local demonstration services where practical, including the necessary MLflow and Airflow components, database or metadata storage, networking, and persistent/shared artifact volumes. The architecture must ensure that the API, Airflow tasks, and training code agree on how artifacts are stored and accessed.

Document how to start and configure a local Kubernetes cluster using an appropriate tool such as Minikube or Kind.

The application must also support offline unit testing without requiring the services to be running.

Do not assume that Docker Compose alone provides a Kubernetes cluster. Explain which components run in Compose and which run in Kubernetes.

### 3.3 Configuration

Support at least the following environment variables, adding others when required:

```text
OPENAI_API_KEY
GEMINI_API_KEY
AGENT_LLM_PROVIDER
MLFLOW_TRACKING_URI
AIRFLOW_API_URL
AIRFLOW_TOKEN
AIRFLOW_USERNAME
AIRFLOW_PASSWORD
AIRFLOW_ALLOW_LEGACY_API
KUBECONFIG
APPROVED_NAMESPACE
MLFLOW_ARTIFACTS_DIR
MLOPS_ARTIFACTS_DIR
MODEL_NAME
MODEL_VERSION
LOG_LEVEL
FRESHNESS_MAX_DAYS
```

Provide sensible local defaults for non-sensitive settings. Do not use a fake credential as a working default.

`APPROVED_NAMESPACE` must be trusted server-side configuration. It must not be selected or overridden by a client request.

Validate and document every required environment variable. Make connection failures understandable without printing credentials.

## 4. PYDANTIC SCHEMAS — `agent/schemas.py`

Implement strict Pydantic v2 models for all externally supplied inputs, tool arguments, tool results, and API responses.

Use strict validation where appropriate, forbid unexpected fields, constrain numeric ranges and string lengths, and avoid accepting arbitrary command strings or unrestricted Kubernetes operations.

At minimum, implement:

### 4.1 ToolResult

Fields:

- `tool_name`
- `status`
- `data`
- `error`
- `duration_ms`
- `request_id`

Allowed status values:

- `SUCCESS`
- `ERROR`
- `TIMEOUT`
- `REJECTED`

Use a consistent error representation. Successful responses must not carry a misleading error, and failed responses must not masquerade as successful operations.

### 4.2 AgentQueryRequest

Include:

- `query: str`
- `request_id: Optional[str]`
- `confirm_cluster_change: bool = False`
- An optional single-use confirmation token for a previously approved, precisely described cluster operation.

Do not expose a client-controlled `approved_namespace` field as an authorization mechanism.

Generate a UUID4 request identifier when one was not supplied. Validate supplied request identifiers and keep the same request ID across all tool calls belonging to the same logical request.

### 4.3 AgentQueryResponse

Include:

- `request_id`
- `answer`
- `known_facts: List[str]`
- `uncertainties: List[str]`
- `recommended_actions: List[str]`
- `tool_traces: List[ToolResult]`

Support an explicit indication when an operation requires confirmation before execution.

### 4.4 Additional schemas

Create appropriate input and output models for dataset operations, MLflow searches, Airflow operations, Kubernetes operations, health endpoints, model version metadata, and prediction requests.

Do not pass unvalidated LLM output directly into tool implementations.

## 5. STRUCTURED LOGGING AND SECRET REDACTION — `agent/logger.py`

Implement a structured JSON logger.

Every agent step and tool invocation must record:

- `timestamp`
- `request_id`
- `tool_name`
- `status`
- `duration_ms`
- `details`

Use ISO 8601 UTC timestamps and UUID4 request IDs. Use a consistent logger interface rather than constructing slightly different JSON log formats in each module.

Measure actual tool-call durations and record the final status, including errors, timeouts, and rejected operations.

Implement both key-based and pattern-based secret redaction for:

- API keys
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `MLFLOW_TRACKING_TOKEN`
- `AIRFLOW_TOKEN`
- Airflow usernames and passwords when sensitive
- Kubernetes credentials, tokens, and kubeconfig contents
- Authorization headers and Bearer tokens
- Passwords, connection strings, and credential-bearing URLs

Replace discovered secrets with `***REDACTED***`.

Scrubbing must occur before log serialization. It must protect nested dictionaries, exception messages where feasible, URL credentials, and tool details. Do not log complete HTTP headers or unrestricted raw request bodies.

Do not log secrets merely because an operation succeeded. Log concise, sanitized, operationally useful evidence.

Write automated tests that prove secret redaction works.

## 6. TASK 1 — DVC DATA MANAGEMENT AGENT

Implement `agent/tools/dvc_tools.py`.

Use DVC and Git through subprocess argument lists or supported interfaces. Every subprocess invocation must use `shell=False`, an explicit timeout no greater than 15 seconds for the required DVC/Git calls, and controlled working directories.

Implement an explicit allow-list of permitted DVC and Git operations. Reject unauthorized subcommands, options, path traversal, out-of-repository paths, and invalid revisions. Do not create a generic method that accepts and executes arbitrary user-supplied shell commands.

Prefer structured metadata inspection and supported commands over brittle parsing of terminal output.

### 6.1 Required tools

1. `list_dvc_datasets(repo_path: str = ".") -> dict`

   - Identify datasets tracked by `.dvc` files and dataset-related stages in `dvc.yaml`.
   - Report their paths, tracking mechanism, and available provenance information.
   - Handle missing files and uninitialized repositories explicitly.

2. `get_current_dataset_version(dataset_path: str = "data/raw_dataset.csv", experiment_or_run_id: Optional[str] = None) -> dict`

   - Retrieve the associated `.dvc` metadata and checksum.
   - Obtain the Git commit when available.
   - Inspect configured DVC remotes without exposing credential-bearing URLs.
   - Retrieve MLflow provenance tags when a run ID is provided and the service is available.
   - Report the dataset version according to the documented provenance convention.
   - Distinguish a recorded version from an uncommitted or modified working copy.

3. `compare_dataset_versions(rev_a: str, rev_b: str, dataset_path: str = "data/raw_dataset.csv") -> dict`

   - Validate both Git revisions.
   - Compare their dataset metadata and checksums.
   - Use supported DVC/Git operations as appropriate.
   - Report changed, unchanged, or unavailable provenance explicitly.
   - Do not change the current working tree merely to compare versions.

4. `detect_data_changes(repo_path: str = ".") -> dict`

   - Inspect `dvc status` and `git status --porcelain`.
   - Distinguish data changes, DVC pipeline changes, and unrelated working-tree changes.
   - Do not label every Git modification as dataset drift.
   - Handle missing DVC remotes or unavailable data without inventing a result.

5. `explain_dataset_reproduction(target_version_or_commit: str, dataset_path: str = "data/raw_dataset.csv", execute: bool = False, confirm_action: bool = False) -> dict`

   - Validate the requested revision and dataset path.
   - Explain the exact steps needed to reconstruct that dataset version.
   - Return individual command argument lists or documented commands, not an unsafe shell chain joined with `&&`.
   - Prefer a temporary worktree or another isolated location for an executed reproduction.
   - Default to explanation/dry-run behavior.
   - Require explicit approval before changing a working tree or performing other state-changing actions.
   - Use the documented command sequence for the pinned DVC version, including retrieval of remote data when necessary.

### 6.2 Provenance conventions

Define the relationship among:

- DVC content checksum
- Git commit
- optional Git tag or human-readable version
- `dvc_dataset_version` MLflow tag
- `dataset_md5` MLflow tag

Never treat these values as interchangeable.

When a formal dataset tag is unavailable, report the recorded checksum-based identity rather than inventing a semantic version such as `v2.1`.

Use one documented convention consistently throughout training, MLflow logging, API responses, tests, and reporting.

### 6.3 Dataset implementation

Create a deterministic classification dataset suitable for the required training pipeline.

Provide `training/generate_dataset.py` with a fixed random seed and a documented schema. Generate a real `data/raw_dataset.csv` containing enough rows and numeric features to support repeatable train/validation splitting and model evaluation.

Use a clearly named binary target column, document the feature columns, and validate row count, missing values, data types, and target distribution.

Generate valid DVC metadata from the actual file using DVC. Do not fabricate the checksum, file size, or cache information.

A remote DVC store must be configurable. The local demonstration must not require unavailable remote credentials to run the basic dataset and training tests.

## 7. TASK 2 — MLFLOW EXPERIMENT INTELLIGENCE AGENT

Implement `agent/tools/mlflow_tools.py`.

Use `mlflow.tracking.MlflowClient` and supported MLflow Python APIs for all experiment and registry operations. Never parse the MLflow web interface.

Use `MLFLOW_TRACKING_URI` from trusted configuration. Handle an inaccessible tracking server using structured errors rather than invented experiment data.

Implement the following five tools.

### 7.1 `list_experiments_and_runs(experiment_name: str = "mlops-agent-experiment", max_results: int = 10) -> dict`

- List matching experiments and recent runs.
- Include run IDs, status, timestamps, parameters, metrics, and relevant tags.
- Validate and bound `max_results`.
- Distinguish an empty experiment from a connection failure.

### 7.2 `get_run_details(run_id: str) -> dict`

Retrieve:

- F1 score
- accuracy
- precision
- recall
- ROC-AUC when recorded
- parameters
- artifact URIs
- run status
- dataset provenance tags
- Git commit and checksum
- quality-gate result
- registered-model linkage when available

Represent missing metrics or tags as unavailable. Do not silently substitute zeros or invented values.

### 7.3 `find_best_run(experiment_name: str = "mlops-agent-experiment", metric: str = "f1_score", top_k: int = 5) -> dict`

- Compare the most recent eligible runs using the specified recorded metric.
- Return a complete comparison table or structured comparison array.
- State the metric direction explicitly and validate supported metric names.
- Exclude runs that do not contain a valid value for the requested metric from ranking, and explain their exclusion.
- Identify the highest-scoring candidate from the actual returned data.
- Inspect recorded quality-gate tags, provenance, precision/recall, training/validation metrics, and latency information when available.
- Treat missing data as unknown.
- Do not claim to have detected latency problems or overfitting without evidence.

The response must explicitly state:

"The highest evaluation score alone does not establish production readiness. Production readiness requires passing the defined quality gate, sufficient recorded provenance, and the relevant operational checks."

### 7.4 `compare_experiments(run_ids: List[str]) -> dict`

- Compare at least two valid run IDs.
- Return side-by-side recorded metrics, hyperparameters, timestamps, provenance, quality-gate results, and artifact locations.
- Validate the number and format of identifiers.
- Explain unavailable comparisons instead of fabricating data.

### 7.5 `get_registered_models(model_name: str = "production-classifier") -> dict`

- Use supported registry client methods.
- Retrieve model versions, linked run IDs, descriptions, tags, and aliases.
- Use aliases and tags as the preferred lifecycle mechanisms.
- Report legacy model stages only when supported, clearly identifying their legacy status.
- Handle models that are not registered.

Do not automatically create a production alias or promote a model merely because it has the highest score.

### 7.6 Model and quality-gate contract

Use documented MLflow APIs compatible with the chosen version.

Model evaluation must log the supported metrics, parameters, model artifact, and valid provenance tags. Record whether the quality gate passed or failed.

When a metric or provenance field does not exist, do not silently treat it as valid. A missing required gate metric or required provenance must cause the relevant gate to fail or be reported as unverified according to the defined policy.

Use one consistent production-readiness policy across training, the MLflow tools, the orchestrator, and tests.

## 8. TASK 3 — APACHE AIRFLOW 3 WORKFLOW AGENT

Implement `agent/tools/airflow_tools.py` and `dags/training_pipeline.py`.

Use authenticated HTTP requests via `httpx` or a compatible supported client to interact with the documented Airflow 3 REST API.

Support:

- `AIRFLOW_API_URL`
- Bearer authentication using `AIRFLOW_TOKEN`
- Basic authentication using environment-configured credentials, where applicable
- connection and read timeouts
- response validation
- bounded retry behavior for transient errors where appropriate
- structured handling of HTTP failures and timeouts

Use the documented Airflow 3 API namespace, normally `/api/v2` for Airflow 3. Do not silently switch between API versions after an authentication failure or an unrelated server error.

If legacy Airflow API compatibility is intentionally implemented, it must be an explicit, documented configuration and must be validated against the corresponding official API documentation.

Implement these five tools.

### 8.1 `list_available_dags() -> dict`

- Retrieve available DAGs.
- Report IDs, pause status, and documented schedule information.
- Handle unavailable or malformed responses.

### 8.2 `get_dag_run_status(dag_id: str = "ml_training_pipeline", dag_run_id: Optional[str] = None) -> dict`

- Retrieve a specific DAG run or the most recent run.
- Return real execution status, timestamps, configuration, and available summary details.
- If no run exists, report that no run was found.

### 8.3 `trigger_training_dag(dag_id: str = "ml_training_pipeline", conf: Optional[dict] = None) -> dict`

- Allow only explicitly approved DAG IDs, initially `ml_training_pipeline`.
- Validate request configuration.
- Trigger the run only in response to an authorized explicit request.
- Return the actual DAG run ID and response status.
- Never trigger training automatically from an informational or diagnostic request.

### 8.4 `identify_failed_tasks(dag_id: str = "ml_training_pipeline", dag_run_id: str = "latest") -> dict`

- Retrieve task-instance information for the selected run.
- Identify actual `failed` and `upstream_failed` states.
- Report unavailable task information explicitly.
- Do not infer the cause of failure solely from a failed state.

### 8.5 `get_task_diagnostics(dag_id: str, dag_run_id: str, task_id: str, try_number: int = 1) -> dict`

- Retrieve supported task metadata, execution attempts, and accessible logs.
- Retrieve documented error/traceback information where the configured API exposes it.
- Handle unavailable logs without fabricating tracebacks.
- Validate task IDs and the requested attempt number.

### 8.6 Required DAG

Create a real Airflow 3 DAG with:

`dag_id="ml_training_pipeline"`

The DAG must contain exactly these five logical stages in this order:

`validate_dataset >> preprocess_data >> train_model >> evaluate_and_log_mlflow >> apply_model_quality_gate`

Configure two retries and a one-minute retry delay where supported by the selected Airflow API. Use structured Python logging and explicit task dependencies.

Use `airflow.sdk` and other supported public Airflow 3 interfaces according to the pinned version's official documentation.

#### Stage 1: `validate_dataset`

- Validate the CSV schema.
- Validate the target column and required features.
- Reject missing, invalid, or inappropriate data types.
- Check minimum row count.
- Check the existence and validity of the applicable DVC metadata.
- Fail using an appropriate Airflow exception when validation fails.

#### Stage 2: `preprocess_data`

- Perform deterministic preprocessing.
- Split training and validation data without leakage.
- Fit preprocessing transformations on training data only.
- Persist reusable transformations and processed artifacts.
- Record paths and metadata needed by downstream stages.

#### Stage 3: `train_model`

- Invoke shared logic implemented in `training/train.py`.
- Use a RandomForestClassifier.
- Set random seeds and relevant hyperparameters.
- Save the trained model and associated metadata using the documented artifact contract.

#### Stage 4: `evaluate_and_log_mlflow`

- Evaluate the model on the held-out validation data.
- Compute F1 score, accuracy, precision, recall, and ROC-AUC when supported by the dataset.
- Log parameters, metrics, model artifact, dataset checksum, dataset version identity, and Git commit when available.
- Record missing provenance and data limitations explicitly.
- Do not register or promote the model before the quality gate succeeds.

#### Stage 5: `apply_model_quality_gate`

Enforce the assignment's minimum thresholds:

- F1 score >= 0.80
- Accuracy >= 0.80
- Valid recorded DVC dataset provenance

Make the metric comparisons and provenance policy explicit. Do not lower thresholds just to obtain a successful run.

If validation fails, raise an appropriate exception and ensure that the quality gate is marked failed.

Only after the gate passes may the workflow register the model or attach an approved candidate alias/tag. Do not automatically assign a production alias or deploy a new model as part of routine training.

### 8.7 Reliable cross-task data flow

Define and implement a consistent artifact contract among tasks. Store necessary data, model files, and metadata in the configured shared artifact directory or an explicitly configured artifact store.

Use XCom for small, serializable metadata such as paths, identifiers, and metric summaries, not large datasets or model binaries.

Avoid reliance on an undocumented local file that happens to exist on one machine. Make task retries safe and outputs deterministic for a given run.

## 9. TASK 4 — KUBERNETES OPERATIONS AGENT

Implement `agent/tools/kubernetes_tools.py` using the official Kubernetes Python client.

Use supported `CoreV1Api` and `AppsV1Api` methods. Load kubeconfig safely for local use and support in-cluster configuration when appropriate. Handle absent kubeconfig, authentication failure, authorization failure, and cluster connection errors.

### 9.1 Non-negotiable Kubernetes authorization

All operations that modify cluster state must be rejected unless both conditions are satisfied:

1. The requested operation has explicit user confirmation.
2. The namespace exactly matches the server-configured `APPROVED_NAMESPACE`, defaulting to `mlops-prod`.

Do not accept the approved namespace from a user-controlled request field. A requested namespace is a target, not a source of authority.

Implement a confirmation workflow:

- An initial request to mutate infrastructure returns a confirmation-required response and describes the precise action.
- A subsequent request must include the explicit confirmation flag and a valid, single-use token bound to the previously described action and namespace.
- The server validates the token, expiration, action, namespace, and operation before allowing the mutation.
- Tokens must not be reusable, and an altered action must invalidate the confirmation.
- If confirmation fails, return a structured `REJECTED` result before contacting Kubernetes for the mutation.

Keep the confirmation mechanism small and testable. A short-lived in-memory implementation is acceptable for this coursework if its limitations across restarts or multiple application replicas are documented. Do not claim distributed or persistent authorization guarantees that the implementation does not provide.

Restrict mutation operations to an explicit allow-list. Do not expose generic deletion, arbitrary manifest application, arbitrary shell execution, or unbounded scaling.

### 9.2 Required tools

Implement these six tools.

1. `list_pods_and_deployments(namespace: str = "mlops-prod") -> dict`

   - List Pods and Deployments.
   - Report observed Pod phases, container readiness, restart counts, desired replicas, ready replicas, and available replicas.
   - Distinguish healthy, unhealthy, pending, and unknown states.
   - Report the namespace inspected.

2. `get_pod_status_and_logs(pod_name: str, namespace: str = "mlops-prod", tail_lines: int = 50) -> dict`

   - Validate Pod name, namespace, and bounded log length.
   - Retrieve status, container state and last termination state, restart counts, and configured readiness/liveness probes where available.
   - Retrieve recent logs only when accessible.
   - Do not claim a probe failed without evidence of its outcome.

3. `inspect_services(namespace: str = "mlops-prod") -> dict`

   - Report Services, types, ClusterIPs, ports, and selectors.
   - Explain whether the observed selectors match expected application Pods where that can be determined from real data.

4. `diagnose_unhealthy_pods(namespace: str = "mlops-prod") -> dict`

   - Inspect Pod/container status and relevant events.
   - Identify evidence of `CrashLoopBackOff`, `OOMKilled`, `ImagePullBackOff`, unsuccessful readiness, repeated restarts, and other actual failures.
   - Correlate logs, events, termination reasons, and Deployment readiness.
   - Present root causes as confirmed, likely, or unknown, with the supporting evidence.
   - Do not invent missing events or logs.

5. `get_deployed_app_version(deployment_name: str = "mlops-inference-deployment", namespace: str = "mlops-prod") -> dict`

   - Read deployed image tags or digests.
   - Inspect application version labels.
   - Read model-version metadata from the relevant ConfigMap or declared metadata source.
   - Return dataset provenance only if it is genuinely available from a recorded deployment metadata source.
   - Distinguish the application version from the model version and dataset version.

6. `scale_or_restart_deployment(deployment_name: str, namespace: str = "mlops-prod", replicas: int = 2, confirm_action: bool = False, confirmation_token: Optional[str] = None, operation: str = "scale") -> dict`

   - Support only explicitly implemented operations, initially `scale` and `restart`.
   - Validate deployment name, namespace, allowed operation, and replica limits.
   - For scaling, enforce a documented range appropriate to this assignment.
   - For restarting, use a supported Deployment operation such as a controlled pod-template annotation update where appropriate.
   - Enforce the full confirmation workflow and server-side namespace lock.
   - Return the observed result of the mutation, not a claim of success based merely on a successful request.
   - Report authorization or API errors accurately.

If the approved namespace is not configured or does not exist, do not silently fall back to `default`, `kube-system`, or another namespace.

### 9.3 Required Kubernetes manifests

Implement the following:

**`kubernetes/namespace.yaml`**

- Define namespace `mlops-prod`.

**`kubernetes/configmap.yaml`**

- Store non-sensitive configuration, including:
  - `MODEL_NAME: production-classifier`
  - `MODEL_VERSION` according to the chosen artifact
  - `LOG_LEVEL: INFO`
  - non-secret MLflow tracking configuration appropriate to the deployment
  - `FRESHNESS_MAX_DAYS: "14"`

Do not store API keys, tokens, or passwords in ConfigMaps.

**`kubernetes/deployment.yaml`**

- Deployment name: `mlops-inference-deployment`
- Namespace: `mlops-prod`
- Desired replicas: 2
- A real, buildable application image
- A consistent image tag strategy
- `envFrom` referencing the non-sensitive ConfigMap
- Readiness probe: `GET /health/ready`
- Liveness probe: `GET /health/live`
- CPU requests: `100m`
- CPU limits: `500m`
- Memory requests: `128Mi`
- Memory limits: `512Mi`

Include safe rolling-update configuration and appropriate container settings. Verify that probe paths and the listening port match the application.

**`kubernetes/service.yaml`**

- ClusterIP Service
- Service port 80
- Target port 8000
- Selector matching the Deployment's Pods

The manifests must be consistent in names, labels, namespace, ports, environment variables, and model-artifact expectations.

## 10. FASTAPI INFERENCE AND AGENT INTERFACE

Implement `api/main.py`.

The FastAPI application must serve both inference and conversational agent requests. Use separate modules and functions for the inference runtime and agent orchestration.

### 10.1 Required endpoints

**`GET /health/live`**

- Report whether the process is alive.
- Liveness must not fail solely because MLflow or Airflow is temporarily unavailable.

**`GET /health/ready`**

- Report readiness based on required local serving prerequisites, including successful model loading when model inference is required.
- Return an appropriate unhealthy HTTP status when the application cannot safely serve predictions.
- Do not report readiness merely because the web server is running.

**`GET /version`**

Return available application version, loaded model version, and recorded dataset provenance. Return explicit unknown/unavailable fields when metadata cannot be established.

**`POST /predict`**

- Accept a validated feature vector matching the training schema.
- Reject the wrong number of features, non-finite numbers, and invalid payloads.
- Load the actual trained model using the declared model-artifact contract.
- Return the real prediction and measured request latency.
- Do not invent prediction probabilities, metrics, or model versions.

**`POST /agent/query`**

- Validate `AgentQueryRequest`.
- Invoke `MLOpsAgentOrchestrator`.
- Return the structured `AgentQueryResponse`.
- Include real tool traces, known facts, uncertainties, and recommendations.
- Support confirmation-required responses for cluster modifications.
- Apply request timeouts and error handling.
- Never return a successful operational answer merely because the LLM generated plausible text.

### 10.2 Model lifecycle

Define one consistent model-artifact contract in training and serving.

The training pipeline must save the actual model artifact and metadata. The serving process must load the intended model using the configured artifact path or an explicitly supported MLflow model URI/alias.

Do not train an untracked model silently during API startup.

When the artifact is unavailable or invalid:

- Return a clear readiness failure.
- Reject predictions with an appropriate response.
- Report the actual error in logs after redaction.
- Avoid falling back silently to a randomly initialized or unrelated model.

Document how the model is made available to the API container and Kubernetes Pods. Do not assume the API container can access a training container's private local filesystem without an explicit volume or artifact-store contract.

## 11. MULTI-TOOL AGENT ORCHESTRATOR

Implement `agent/orchestrator.py` with a class named `MLOpsAgentOrchestrator`.

The orchestrator must combine the four operational modules into a single agent.

### 11.1 LLM and offline operation

Support two execution modes:

**LLM-assisted mode**

- Support OpenAI function/tool calling using `OPENAI_API_KEY`.
- Support Gemini function/tool calling using `GEMINI_API_KEY` and the `google-genai` SDK if both providers are implemented.
- Choose the provider through `AGENT_LLM_PROVIDER`.
- Validate all tool arguments through Pydantic schemas.
- Expose only the approved tools.
- Do not permit an LLM to bypass authorization controls.
- Never pass API keys or other secrets into prompts or tool arguments.

**Deterministic offline mode**

- Work without any LLM API key.
- Use an explicit, rule-aware planner to identify which tools a request requires.
- Support tool routing based on clear intents and entities.
- Execute real locally available tool functions where the relevant service is configured.
- Return honest connection errors when external services are unavailable.

The offline planner must support the same mandatory multi-tool workflow as the LLM-assisted mode. Tests must not require internet access or live API credentials.

If a configured LLM provider is unavailable, do not conceal the failure. A documented fallback to the deterministic planner is acceptable, provided that the response records the provider failure and does not invent operational results.

### 11.2 Tool execution and response generation

- Give every invocation structured, validated inputs and outputs.
- Preserve the request ID through all tool calls.
- Record status and duration for each tool.
- Catch exceptions at appropriate boundaries.
- Enforce bounded execution time and avoid unbounded retries.
- Distinguish facts obtained from tools from inferences and recommendations.
- Do not mark a request successful when the required evidence could not be retrieved.
- If tools disagree or return stale or incomplete information, report the discrepancy.
- Avoid invoking unrelated tools when they are not needed to answer the question.

### 11.3 Mandatory integration scenario

The orchestrator must support the following request:

"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."

It must:

1. Query MLflow to retrieve recent runs, relevant metrics, the best eligible candidate, and recorded quality-gate results.
2. Retrieve the selected run's DVC provenance, including dataset-version identity, dataset checksum, and Git commit when recorded.
3. Inspect DVC status and current dataset changes.
4. Inspect the Kubernetes deployment version and diagnose unhealthy Pods.
5. Retrieve recent Airflow DAG-run status and failed tasks when relevant.
6. Correlate the actual observations.
7. Populate `known_facts`, `uncertainties`, and `recommended_actions`.
8. Identify the sources supporting each operational conclusion.

If a service is unavailable, the agent must still provide useful results from the remaining available services while explicitly listing what it could not determine.

### 11.4 Strict retraining policy

Never trigger `trigger_training_dag` automatically from an informational, diagnostic, or model-selection request.

Recommend retraining only when evidence supports it, such as:

- a recorded freshness-policy breach;
- relevant dataset drift;
- a failed quality gate;
- measured model degradation;
- a documented version mismatch or reproducibility problem.

A dataset change alone does not prove that the deployed model has degraded. Explain why the observed evidence supports or does not support retraining.

Triggering a pipeline must require a separately expressed, authorized training request. Cluster mutations must follow their independent confirmation workflow.

## 12. API AND CLI ENTRY POINTS

### 12.1 `agent/main.py`

Implement a functional CLI for interacting with the orchestrator.

Include commands or options for:

- entering an arbitrary natural-language query;
- running the five required evaluation scenarios;
- running the two failure investigations;
- choosing offline or configured LLM-assisted mode;
- printing structured tool traces;
- saving sanitized evidence to an output directory.

Provide useful help text, input validation, proper exit codes, and readable errors.

Do not fabricate live-service output when a service is unavailable.

### 12.2 CLI evidence modes

Support clearly distinct modes where appropriate:

- **Live mode:** query configured services and record the actual returned evidence.
- **Mocked demonstration/test mode:** use explicit test fixtures and label all results as simulated or mocked.

Do not represent mocked results as actual cluster, MLflow, or Airflow observations.

## 13. DELIBERATELY INJECTED FAILURE SIMULATOR

Implement `agent/failure_simulator.py`.

Provide a working demonstration mechanism for both required failure investigations.

### 13.1 Failure 1 — Data quality gate or pipeline failure

Create a controlled scenario in which an injected test dataset problem or recorded sub-threshold result causes the quality gate to fail. The example metric may be F1 = 0.62 against the required threshold of 0.80.

The demonstration must show:

1. The injected failure condition.
2. The actual task or controlled fixture representing the failed quality gate.
3. The real or explicitly mocked task status.
4. The recorded metric and required threshold.
5. The agent's diagnosis and recommended resolution.

Do not silently modify the canonical dataset or corrupt the user's real data. Keep simulation artifacts isolated and restore any temporary changes.

### 13.2 Failure 2 — Kubernetes infrastructure failure

Simulate a controlled example of `CrashLoopBackOff`, a failed readiness condition, an invalid model-artifact path, or an `OOMKilled` event.

Demonstrate how the agent uses Pod status, restart counts, events, and logs to reach a diagnosis.

Default to mocked Kubernetes objects in tests or a clearly isolated demonstration namespace.

Never deliberately damage the `mlops-prod` namespace to create a demonstration failure. Any live fault injection must require a separately configured, isolated environment and explicit authorization.

### 13.3 Failure simulation requirements

- Make the scenarios deterministic and repeatable.
- Label simulated results explicitly.
- Reuse the production diagnostic logic where practical.
- Test the actual diagnosis behavior, not merely that a simulator returns a hardcoded explanatory string.
- Record the root cause, supporting evidence, and appropriate corrective action.

## 14. AUTOMATED TEST SUITE

Implement at least ten meaningful tests in `tests/test_agent_and_tools.py`. Split tests into additional modules if this improves maintainability.

All ordinary CI tests must run offline, using pytest, `unittest.mock`, `monkeypatch`, fake HTTP responses, and controlled Kubernetes/MLflow fixtures as necessary.

### Mandatory tests

**1. `test_dvc_tools_provenance_and_command_allowlist`**

- Validate actual DVC metadata handling.
- Test a temporary Git/DVC repository where feasible.
- Verify valid provenance extraction.
- Verify rejection of unauthorized commands, invalid revisions, path traversal, and shell-style command injection.
- Verify that subprocesses are not invoked with `shell=True`.

**2. `test_mlflow_best_run_selection_and_caveat`**

- Mock recorded MLflow runs.
- Verify ranking by actual F1 scores.
- Verify missing metrics are handled correctly.
- Verify that the output distinguishes score ranking from production readiness.

**3. `test_airflow_dag_structure_and_trigger`**

- Verify all five required task IDs and their exact dependency order.
- Validate retry configuration.
- Mock the Airflow 3 API response.
- Verify triggering occurs only for the approved DAG ID.

**4. `test_kubernetes_inspection_and_diagnosis`**

- Provide mocked Pod, Deployment, event, and log responses.
- Verify that observed `CrashLoopBackOff` or another injected failure is diagnosed using supporting evidence.
- Verify that healthy Pods are not incorrectly reported as unhealthy.

**5. `test_unauthorized_kubernetes_modification_rejected`**

Verify that a mutation is rejected when:

- confirmation is false;
- confirmation token is missing, expired, invalid, or reused;
- the confirmed operation differs from the requested operation;
- the target namespace is unapproved;
- the operation is outside the allow-list.

Assert that the Kubernetes mutation API was not called.

**6. `test_agent_does_not_hallucinate_when_tool_unavailable`**

Mock MLflow and Kubernetes connection failures. Verify that the agent reports service unavailability in `uncertainties` and does not fabricate metrics, Pod names, model versions, or deployment statuses.

**7. `test_tool_timeout_and_invalid_request_handling`**

Test subprocess timeouts, HTTP timeouts, malformed responses, invalid arguments, and safe error reporting.

**8. `test_multi_tool_mandatory_integration_scenario`**

Verify that the deterministic planner invokes the relevant MLflow, DVC, Kubernetes, and Airflow tools and correlates their actual mocked responses. Confirm that `trigger_training_dag` is not called for the informational scenario.

**9. `test_structured_logging_and_secret_redaction`**

Verify the presence of `tool_name`, `status`, `duration_ms`, and `request_id`. Ensure tokens such as `Bearer secret-token-123` are replaced with `***REDACTED***`.

**10. `test_injected_failures_diagnosis`**

Verify both controlled failure investigations and their evidence-based diagnostic outcomes.

### Additional tests

Also test:

- API health and readiness behavior;
- invalid prediction payloads;
- model loading and missing artifacts;
- MLflow connection failures;
- missing dataset provenance;
- Airflow authentication and permission errors;
- Kubernetes configuration errors;
- malformed LLM tool arguments;
- retraining policy;
- confirmation-token expiration and single-use behavior;
- request-ID propagation;
- API response schema validation.

Tests must assert meaningful behavior. Do not weaken assertions, skip failed tests indiscriminately, or change a production threshold solely to make tests pass.

Use mocks for external-service calls, not as hidden replacements for the production implementation.

## 15. GITHUB ACTIONS CI

Implement `.github/workflows/ci.yml`.

Run on pushes and pull requests.

The workflow must:

1. Check out the repository.
2. Set up Python 3.11 for the application.
3. Install the locked application dependencies.
4. Validate the lock file and dependency configuration.
5. Run `ruff check .`.
6. Run `pytest -v`.
7. Keep ordinary tests independent of externally running services and credentials.
8. Publish useful failure output.

If Airflow has a separate runtime environment, validate DAG syntax and structure using the selected version's appropriate environment or a dedicated compatible CI step.

Do not expose GitHub secrets through test output. Do not run real cluster mutations or live training jobs in ordinary CI.

Keep the workflow aligned with the actual repository's dependency and test setup. Do not create a CI workflow that references nonexistent files or scripts.

## 16. DOCKERFILE AND KUBERNETES DEPLOYMENT

Create a buildable Dockerfile for the FastAPI agent and inference service.

Requirements:

- Use a suitable Python 3.11 base image.
- Install only the runtime dependencies needed by the application.
- Run under a non-root user.
- Expose port 8000.
- Launch the correct FastAPI application object.
- Support configuration via environment variables.
- Avoid embedding credentials, model binaries, local DVC caches, or unnecessary development tools.
- Implement appropriate container signal handling.
- Use a build process that is compatible with the documented local and Kubernetes deployment workflows.

Ensure the image tag referenced in the Kubernetes Deployment can actually be built and loaded or pulled by the documented cluster workflow.

Document the commands to build the image, make it available to the cluster, apply the namespace and manifests, inspect Pods, inspect Services, and test health endpoints.

The deployment documentation must also explain artifact availability. A container that starts successfully but cannot load its required model must not be described as ready to serve predictions.

## 17. DVC PIPELINE AND SHARED TRAINING LOGIC

Implement `dvc.yaml` and `training/train.py`.

Provide a reproducible DVC pipeline appropriate to the dataset and model.

- Use explicit dependencies, parameters, outputs, and metrics files.
- Use deterministic processing and random seeds.
- Use the same core preprocessing and training logic from the DVC workflow and Airflow tasks wherever practical.
- Keep the DVC pipeline and Airflow pipeline consistent without making one depend on undocumented files created by the other.
- Define artifact paths and metadata clearly.
- Ensure the training command can execute from a clean, documented environment.
- Do not claim a DVC stage is reproducible if its inputs, outputs, or environment are unspecified.

The Airflow pipeline orchestrates the required five stages. DVC provides a reproducible data/model workflow. Their roles should complement each other rather than implement conflicting versions of the same transformations.

## 18. README AND ARCHITECTURE DOCUMENTATION

Write a comprehensive `README.md` and `architecture.md` suitable for evaluation and a technical defense.

Do not provide generic installation instructions that omit the actual configuration or commands used by this project.

### 18.1 README requirements

Explain:

- assignment objective and system capabilities;
- full repository layout;
- Python environment setup;
- dependency installation and locking;
- environment-variable configuration;
- deterministic dataset generation and DVC tracking;
- MLflow tracking-server setup;
- Airflow 3 setup, DAG placement, API authentication, and DAG execution;
- local Kubernetes cluster setup;
- Docker image build and deployment;
- service startup and shutdown;
- FastAPI endpoints and example payloads;
- CLI commands and agent interactions;
- test execution;
- lint execution;
- failure-simulation commands;
- evidence-generation commands;
- troubleshooting and common failure modes.

Commands must correspond to the actual files and scripts that exist.

### 18.2 Architecture requirements

Include both Mermaid and ASCII architecture diagrams.

Explain the flow among:

- user or API client;
- FastAPI;
- orchestrator and deterministic/LLM planner;
- DVC tool;
- MLflow tool;
- Airflow tool and DAG;
- training pipeline;
- artifact storage and model serving;
- Kubernetes tool and deployed inference service;
- structured logging and error handling.

Explain why each technology is used and what its limits are.

### 18.3 Permission and safety matrix

Document every tool and classify it as read-only, execution-triggering, or state-mutating.

Include authorization conditions, expected failures, and confirmation requirements.

Explicitly distinguish dataset inspection from dataset checkout, Airflow status inspection from pipeline triggering, and Kubernetes diagnostics from cluster modifications.

### 18.4 Five required evaluation scenarios

Provide runnable scenarios for:

1. Dataset provenance.
2. Model selection.
3. Pipeline execution.
4. Deployment diagnosis.
5. Multi-tool reasoning.

For each scenario, document the query, tools expected to execute, actual observed outputs, and evidence supporting the final answer.

Generate transcripts or exported evidence from real executions when the relevant services are available.

If only mocks are available, label the output as mocked. Do not invent a live transcript, screenshot, metric, run status, Pod status, or service response merely to make the report appear complete.

### 18.5 Two failure investigations

Document:

1. The dataset or model-quality-gate failure.
2. The Kubernetes infrastructure failure.

For each, state the initial condition, diagnostic steps, evidence, root cause, resolution, and test or command used to verify the resolution.

Clearly identify whether a result was live or simulated.

### 18.6 Limitations and improvements

Explain known limitations, including unavailable services, artifact-store assumptions, local confirmation-state persistence, the limits of deterministic natural-language routing, and any integration behavior that has not been live-tested.

Suggest appropriate production improvements without claiming they have already been implemented.

## 19. EVIDENCE AND SUBMISSION SAFETY

Implement `screenshots/README.md` with instructions for capturing the five required scenarios and two failure investigations.

Evidence must originate from actual executions or explicitly identified simulated tests.

Never generate fake UI screenshots, invent Airflow histories, fabricate MLflow run IDs, create imaginary Kubernetes logs, or present mocked data as live output.

Before completion, check the repository for credentials and sensitive material. Never submit:

- API keys;
- Kubernetes credentials;
- service-account tokens;
- passwords;
- private kubeconfigs;
- credential-bearing URLs;
- unredacted sensitive operational logs.

Use `.gitignore` and `.env.example` appropriately. Ensure that `.env.example` contains configuration names and safe sample values, never actual secrets.

## 20. IMPLEMENTATION AND VERIFICATION WORKFLOW

Execute the following workflow in the current VS Code workspace.

### Phase A — Inspect and plan

- Inspect the current files, Python version, Git status, and available tooling.
- Identify dependency and API compatibility issues.
- Establish a consistent schema, artifact, provenance, configuration, and authorization contract before implementing the dependent modules.
- Use reasonable documented defaults instead of asking unnecessary clarification questions.

### Phase B — Build the application

Implement all required files and supporting modules.

Prioritize shared interfaces and contracts so that the modules agree on tool results, errors, request IDs, dataset metadata, artifact paths, and permissions.

Avoid duplicated business logic and circular imports.

### Phase C — Configure the reproducible environment

- Generate and verify the deterministic dataset.
- Create valid DVC metadata.
- Create the dependency lock file or documented constrained installation.
- Configure local service connections.
- Verify all manifests and environment-variable names agree with the code.
- Ensure no secret values are written to the repository.

### Phase D — Run automated verification

Run the following commands from the actual repository root:

```bash
ruff check .
pytest -v
```

Also run applicable checks for:

- import and syntax validity;
- configuration parsing;
- DVC metadata consistency;
- DAG structure;
- Dockerfile and manifest consistency;
- Pydantic request/response contracts;
- tests of command and permission allow-lists.

If a required command fails, investigate the actual failure, correct the code or configuration, and rerun the relevant checks.

Do not suppress errors simply to obtain a successful exit status. Do not claim a clean lint or test run unless it has actually completed successfully.

### Phase E — Verify local behavior

Run all feasible local checks, including the CLI, API health endpoints, valid and invalid prediction payloads, deterministic planner routing, tool timeout handling, and injected failure diagnoses.

Verify that an unavailable external service results in a truthful error instead of invented data.

### Phase F — Verify service integrations

When the necessary services and credentials are available, run live checks for DVC, MLflow, Airflow 3, and Kubernetes.

Verify real dataset provenance, a real MLflow run, a real Airflow DAG execution, and the observed status of the deployed inference service.

Do not run destructive or state-changing demonstrations against production infrastructure.

If live integration is not possible, complete all feasible mocked tests and report the unverified live checks.

### Phase G — Prepare the submission report

Update the README and architecture documentation based on the implementation that actually exists.

Record the actual lint and test results, the scenarios demonstrated, whether each result is live or mocked, and all unresolved issues.

Do not describe a planned feature as implemented or a mocked interaction as an actual external service execution.

## 21. FINAL ACCEPTANCE CRITERIA

The work is complete only to the extent supported by the code and evidence.

Assess it against these gates:

**Gate 1 — Repository integrity**

Every required file exists, all imports resolve in the intended environment, the documented commands match actual files, and no essential implementation is a stub.

**Gate 2 — Reproducibility**

The dataset, DVC metadata, training parameters, model artifacts, and provenance conventions are consistent and reproducible through the documented workflow.

**Gate 3 — Four operational modules**

DVC, MLflow, Airflow, and Kubernetes each have the required callable tools, structured results, error handling, and automated coverage.

**Gate 4 — Agent integration**

The deterministic planner can orchestrate the mandatory four-technology integration scenario, correlate retrieved evidence, and report uncertainties.

**Gate 5 — Safety**

Unauthorized cluster modifications are rejected; reproduction or training actions are not triggered by unrelated informational requests; subprocess operations are restricted; and secrets are redacted.

**Gate 6 — Reliability**

Timeouts, unavailable services, invalid requests, and injected failures produce meaningful, structured results without fabricated operational claims.

**Gate 7 — Automated quality**

Ruff and pytest have been executed and their actual results recorded.

**Gate 8 — Demonstration and documentation**

All five evaluation scenarios and both failure investigations have runnable procedures. Evidence is captured when feasible and accurately labelled as live, mocked, or not run.

**Gate 9 — Operational honesty**

The final report distinguishes implemented functionality, passing tests, successful live integrations, simulated behavior, and outstanding blockers.

## 22. REQUIRED FINAL RESPONSE

After implementation and verification, provide a concise engineering completion report containing:

1. The repository path and major files created or updated.
2. The capabilities implemented in each of the four MLOps modules.
3. Dependency and infrastructure setup instructions.
4. The exact Ruff command result.
5. The exact pytest command result and number of tests passed or failed.
6. The live integration checks actually completed.
7. The five evaluation scenarios and their evidence status.
8. Both failure investigations and their evidence status.
9. Any known limitations, failed checks, missing services, or unverified requirements.
10. The exact commands required to reproduce the working demonstration.

Never mark a verification gate as passed without supporting evidence.

Begin by inspecting the current workspace. Then implement the complete system, run the required checks, fix the discovered issues, and produce the final report based on the results you actually observe.py -0p