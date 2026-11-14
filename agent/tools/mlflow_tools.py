"""MLflow tracking and registry tools with honest unavailable-service results."""

from __future__ import annotations

import os
from typing import Any

QUALITY_CAVEAT = "The highest evaluation score alone does not establish production readiness. Production readiness requires passing the defined quality gate, sufficient recorded provenance, and the relevant operational checks."
METRICS = {"f1_score", "accuracy", "precision", "recall", "roc_auc"}


def _client() -> Any:
    from mlflow.tracking import MlflowClient
    return MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))


def _error(name: str, exc: Exception) -> dict[str, Any]:
    return {"tool": name, "available": False, "error": f"{type(exc).__name__}: {exc}"}


def list_experiments_and_runs(experiment_name: str = "mlops-agent-experiment", max_results: int = 10) -> dict[str, Any]:
    if not 1 <= max_results <= 100:
        raise ValueError("max_results must be between 1 and 100")
    try:
        client = _client()
        experiments = client.search_experiments(filter_string=f"name = '{experiment_name}'")
        runs = []
        for experiment in experiments:
            runs.extend(client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=max_results))
        return {"available": True, "experiments": [{"id": item.experiment_id, "name": item.name} for item in experiments], "runs": [_run_dict(run) for run in runs[:max_results]]}
    except Exception as exc:
        return _error("list_experiments_and_runs", exc)


def _run_dict(run: Any) -> dict[str, Any]:
    return {"run_id": run.info.run_id, "status": run.info.status, "start_time": run.info.start_time, "end_time": run.info.end_time, "params": dict(run.data.params), "metrics": dict(run.data.metrics), "tags": dict(run.data.tags), "artifact_uri": run.info.artifact_uri}


def _artifact_paths(client: Any, run_id: str) -> list[str]:
    paths: list[str] = []

    def visit(path: str = "") -> None:
        for item in client.list_artifacts(run_id, path):
            if item.is_dir:
                visit(item.path)
            else:
                paths.append(item.path)

    visit()
    return sorted(paths)


def get_run_details(run_id: str) -> dict[str, Any]:
    if not run_id or len(run_id) > 100:
        raise ValueError("invalid run_id")
    try:
        run = _client().get_run(run_id)
        return {
            "available": True,
            **_run_dict(run),
            "artifacts": _artifact_paths(_client(), run_id),
            "provenance": {key: run.data.tags.get(key) for key in ("dvc_dataset_version", "dataset_md5", "git_commit")},
            "quality_gate": run.data.tags.get("quality_gate"),
        }
    except Exception as exc:
        return _error("get_run_details", exc)


def find_best_run(experiment_name: str = "mlops-agent-experiment", metric: str = "f1_score", top_k: int = 5) -> dict[str, Any]:
    if metric not in METRICS or not 1 <= top_k <= 100:
        raise ValueError("unsupported metric or top_k")
    result = list_experiments_and_runs(experiment_name, top_k)
    if not result.get("available"):
        result["caveat"] = QUALITY_CAVEAT
        return result
    eligible, excluded = [], []
    for run in result["runs"]:
        value = run["metrics"].get(metric)
        if isinstance(value, (int, float)):
            eligible.append({**run, "score": value})
        else:
            excluded.append({"run_id": run["run_id"], "reason": f"missing {metric}"})
    eligible.sort(key=lambda item: item["score"], reverse=True)
    return {"available": True, "metric": metric, "direction": "higher_is_better", "best_run": eligible[0] if eligible else None, "comparison": eligible, "excluded": excluded, "caveat": QUALITY_CAVEAT}


def compare_experiments(run_ids: list[str]) -> dict[str, Any]:
    if len(run_ids) < 2 or len(run_ids) > 20 or any(not item or len(item) > 100 for item in run_ids):
        raise ValueError("provide between two and twenty valid run IDs")
    return {"runs": [get_run_details(run_id) for run_id in run_ids]}


def get_registered_models(model_name: str = "production-classifier") -> dict[str, Any]:
    if not model_name or len(model_name) > 120:
        raise ValueError("invalid model name")
    try:
        versions = _client().search_model_versions(f"name='{model_name}'")
        return {"available": True, "model_name": model_name, "versions": [{"version": v.version, "run_id": v.run_id, "description": v.description, "tags": dict(v.tags), "aliases": list(getattr(v, "aliases", []))} for v in versions]}
    except Exception as exc:
        return _error("get_registered_models", exc)

