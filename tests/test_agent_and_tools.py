from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from agent.failure_simulator import simulate_kubernetes_failure, simulate_quality_gate_failure
from agent.logger import REDACTED, redact
from agent.orchestrator import MLOpsAgentOrchestrator
from agent.schemas import AgentQueryRequest
from agent.tools import airflow_tools, dvc_tools, kubernetes_tools, mlflow_tools
from training.generate_dataset import generate
from training.train import train_and_evaluate


def test_secret_redaction_and_schema():
    assert REDACTED in redact("Authorization: Bearer secret-token-123")
    request = AgentQueryRequest(query="status")
    assert request.request_id


def test_best_run_selection_and_caveat():
    runs = {"available": True, "runs": [
        {"run_id": "a", "metrics": {"f1_score": 0.81}},
        {"run_id": "b", "metrics": {}},
        {"run_id": "c", "metrics": {"f1_score": 0.91}},
    ]}
    with patch.object(mlflow_tools, "list_experiments_and_runs", return_value=runs):
        result = mlflow_tools.find_best_run()
    assert result["best_run"]["run_id"] == "c"
    assert "highest evaluation score alone" in result["caveat"]
    assert result["excluded"][0]["run_id"] == "b"


def test_airflow_only_approves_training_dag():
    result = airflow_tools.trigger_training_dag("other_dag")
    assert result["status"] == "REJECTED"


def test_kubernetes_confirmation_is_single_use_and_bound():
    fake_api = Mock()
    with patch.object(kubernetes_tools, "_apis", return_value=(Mock(), fake_api)), patch.dict("os.environ", {"APPROVED_NAMESPACE": "mlops-prod"}):
        first = kubernetes_tools.scale_or_restart_deployment("app", confirm_action=False)
        assert first["confirmation_required"]
        second = kubernetes_tools.scale_or_restart_deployment("app", confirm_action=True, confirmation_token=first["confirmation_token"])
        assert second["status"] == "SUCCESS"
        third = kubernetes_tools.scale_or_restart_deployment("app", confirm_action=True, confirmation_token=first["confirmation_token"])
        assert third["status"] == "REJECTED"
        fake_api.patch_namespaced_deployment_scale.assert_called_once()


def test_orchestrator_reports_unavailable_services_without_fabrication():
    tools = {name: Mock(return_value={"available": False, "error": "connection failed"}) for name in ("best_model", "dataset_version", "data_changes", "deployment_version", "pod_diagnosis", "dag_status", "failed_tasks")}
    response = MLOpsAgentOrchestrator(tools).query("find the best recent model and check deployment status", uuid4())
    assert response.uncertainties
    assert "run_id" not in response.answer
    assert not any("pod-" in fact for fact in response.known_facts)


def test_failure_simulators_are_explicitly_simulated():
    assert simulate_quality_gate_failure()["simulated"] is True
    assert simulate_kubernetes_failure()["simulated"] is True


def test_invalid_airflow_task_request():
    with pytest.raises(ValueError):
        airflow_tools.get_task_diagnostics("bad/id", "run", "task")


def test_api_readiness_rejects_missing_model(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_ARTIFACT_PATH", "missing/model.joblib")
    response = TestClient(__import__("api.main", fromlist=["app"]).app).get("/health/ready")
    assert response.status_code == 503


def test_api_rejects_wrong_feature_count():
    from fastapi.testclient import TestClient

    response = TestClient(__import__("api.main", fromlist=["app"]).app).post("/predict", json={"features": [1, 2]})
    assert response.status_code == 422


def test_kubernetes_namespace_is_server_locked():
    result = kubernetes_tools.scale_or_restart_deployment("app", namespace="default")
    assert result["status"] == "REJECTED"


def test_kubernetes_mutation_requires_configured_namespace(monkeypatch):
    monkeypatch.delenv("APPROVED_NAMESPACE", raising=False)
    result = kubernetes_tools.scale_or_restart_deployment("app", namespace="mlops-prod")
    assert result["status"] == "REJECTED"


def test_training_persists_fitted_preprocessor_and_uses_it(tmp_path):
    dataset = generate(tmp_path / "dataset.csv", rows=200)
    result = train_and_evaluate(str(dataset), str(tmp_path / "artifacts"))
    assert result["quality_gate"] is True
    assert (tmp_path / "artifacts" / "preprocessor.joblib").is_file()
    assert (tmp_path / "artifacts" / "processed_data.joblib").is_file()
    assert result["preprocessor_path"].endswith("preprocessor.joblib")


def test_airflow_preprocess_creates_small_artifact_metadata(tmp_path, monkeypatch):
    dataset = generate(tmp_path / "dataset.csv", rows=200)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    dataset.replace(tmp_path / "data" / "raw_dataset.csv")
    from dags.training_pipeline import preprocess_data

    result = preprocess_data()
    assert result["train_rows"] == 160
    assert result["validation_rows"] == 40
    assert (tmp_path / "artifacts" / "preprocessor.joblib").is_file()


def test_dvc_reproduction_requires_confirmation_and_isolated_git():
    rejected = dvc_tools.explain_dataset_reproduction("HEAD", execute=True)
    assert rejected["status"] == "REJECTED"
    result = dvc_tools.explain_dataset_reproduction("HEAD", execute=True, confirm_action=True)
    assert result["status"] == "ERROR"
    assert "Git repository" in result["error"]


def test_dvc_metadata_is_structured_and_remote_names_are_safe(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "raw.csv").write_text("feature,target\n1,0\n", encoding="utf-8")
    (data_dir / "raw.csv.dvc").write_text(
        "outs:\n- md5: abc123\n  size: 15\n  path: raw.csv\n",
        encoding="utf-8",
    )
    (tmp_path / "dvc.yaml").write_text("stages:\n  train:\n    cmd: python train.py\n", encoding="utf-8")
    result = dvc_tools.list_dvc_datasets(str(tmp_path))
    metadata = dvc_tools._metadata(tmp_path, "data/raw.csv")
    assert result["pipeline_stages"] == ["train"]
    assert result["datasets"][0]["tracking"] == "DVC"
    assert metadata["checksum"] == "abc123"
    assert metadata["size"] == 15


def test_airflow_response_validation_and_timeout():
    response = Mock(status_code=200, headers={"content-type": "application/json"})
    response.raise_for_status.return_value = None
    response.json.return_value = {"unexpected": []}
    with patch("agent.tools.airflow_tools.httpx.request", return_value=response):
        result = airflow_tools.list_available_dags()
    assert result["available"] is False
    assert "dags list" in result["error"]

    with patch(
        "agent.tools.airflow_tools.httpx.request",
        side_effect=__import__("httpx").ReadTimeout("timed out"),
    ):
        result = airflow_tools.list_available_dags()
    assert result["status"] == "TIMEOUT"


def test_kubernetes_diagnosis_uses_observed_failure_evidence():
    status = SimpleNamespace(
        phase="Running",
        container_statuses=[
            SimpleNamespace(
                name="inference",
                ready=False,
                restart_count=5,
                state=SimpleNamespace(reason="CrashLoopBackOff"),
                last_state=SimpleNamespace(reason="OOMKilled"),
            )
        ],
    )
    pod = SimpleNamespace(metadata=SimpleNamespace(name="inference-1"), status=status)
    warning = SimpleNamespace(reason="BackOff", message="container repeatedly restarted", type="Warning")
    core = Mock()
    core.list_namespaced_pod.return_value.items = [pod]
    core.list_namespaced_event.return_value.items = [warning]
    apps = Mock()
    deployment = SimpleNamespace(
        metadata=SimpleNamespace(name="inference"),
        spec=SimpleNamespace(replicas=2),
        status=SimpleNamespace(ready_replicas=0, available_replicas=0),
    )
    apps.list_namespaced_deployment.return_value.items = [deployment]
    with patch.object(kubernetes_tools, "_apis", return_value=(core, apps)):
        result = kubernetes_tools.diagnose_unhealthy_pods()
    assert result["healthy"] is False
    assert "CrashLoopBackOff" in result["findings"][0]["evidence"]["reasons"]
    assert result["findings"][0]["evidence"]["warning_events"]


def test_confirmation_expiry_rejects_before_kubernetes_call(monkeypatch):
    monkeypatch.setenv("APPROVED_NAMESPACE", "mlops-prod")
    token = kubernetes_tools.confirmations.issue("scale", "mlops-prod", ttl=-1)
    fake_apps = Mock()
    with patch.object(kubernetes_tools, "_apis", return_value=(Mock(), fake_apps)):
        result = kubernetes_tools.scale_or_restart_deployment(
            "app",
            confirm_action=True,
            confirmation_token=token,
        )
    assert result["status"] == "REJECTED"
    fake_apps.patch_namespaced_deployment_scale.assert_not_called()


def test_orchestrator_preserves_request_id_and_correlates_selected_run():
    request_id = uuid4()
    calls = []

    def best_model(**_: object):
        calls.append("best_model")
        return {"available": True, "best_run": {"run_id": "run-1"}}

    def dataset_version(**kwargs: object):
        calls.append(("dataset_version", kwargs))
        return {"available": True, "provenance": {"checksum": "abc"}}

    tools = {
        "best_model": best_model,
        "dataset_version": dataset_version,
        "data_changes": lambda **_: {"available": True},
        "deployment_version": lambda **_: {"available": True},
        "pod_diagnosis": lambda **_: {"available": True},
        "dag_status": lambda **_: {"available": True},
        "failed_tasks": lambda **_: {"available": True},
        "run_details": lambda **_: {"available": True, "provenance": {"dataset_md5": "abc"}},
    }
    response = MLOpsAgentOrchestrator(tools).query("find the best recent model", request_id)
    assert response.request_id == request_id
    assert ("dataset_version", {"experiment_or_run_id": "run-1"}) in calls
    assert all(trace.request_id == request_id for trace in response.tool_traces)

