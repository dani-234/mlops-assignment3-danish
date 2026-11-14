"""Airflow 3 REST API client."""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

_ID = re.compile(r"^[A-Za-z0-9_.-]{1,200}$")


class AirflowClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("AIRFLOW_API_URL", "http://localhost:8080/api/v2").rstrip("/")
        self.timeout = httpx.Timeout(10.0, connect=3.0)

    def _request(self, method: str, path: str, allow_text: bool = False, **kwargs: Any) -> dict[str, Any]:
        headers = {}
        if token := os.getenv("AIRFLOW_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"
        auth = None
        if not token and os.getenv("AIRFLOW_USERNAME"):
            auth = (os.environ["AIRFLOW_USERNAME"], os.getenv("AIRFLOW_PASSWORD", ""))
        try:
            response = httpx.request(method, self.base_url + path, headers=headers, auth=auth, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            if allow_text and "application/json" not in response.headers.get("content-type", ""):
                payload = {"logs": response.text}
            else:
                payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Airflow returned a non-object response")
            return {"available": True, "response": payload, "http_status": response.status_code}
        except httpx.TimeoutException as exc:
            return {"available": False, "status": "TIMEOUT", "error": f"{type(exc).__name__}: {exc}"}
        except (httpx.HTTPError, ValueError) as exc:
            return {"available": False, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}


def list_available_dags() -> dict[str, Any]:
    result = AirflowClient()._request("GET", "/dags")
    if not result.get("available"):
        return result
    dags = result["response"].get("dags")
    if not isinstance(dags, list):
        return {"available": False, "status": "ERROR", "error": "Airflow response has no valid dags list"}
    return {
        **result,
        "dags": [
            {
                "dag_id": item.get("dag_id"),
                "is_paused": item.get("is_paused"),
                "schedule": item.get("schedule"),
            }
            for item in dags
            if isinstance(item, dict)
        ],
    }


def get_dag_run_status(dag_id: str = "ml_training_pipeline", dag_run_id: str | None = None) -> dict[str, Any]:
    if not _ID.fullmatch(dag_id) or dag_run_id is not None and not _ID.fullmatch(dag_run_id):
        raise ValueError("invalid DAG identifier")
    path = f"/dags/{dag_id}/dagRuns/{dag_run_id}" if dag_run_id else f"/dags/{dag_id}/dagRuns?limit=1&order_by=-logical_date"
    result = AirflowClient()._request("GET", path)
    if not result.get("available") or dag_run_id:
        return result
    runs = result["response"].get("dag_runs")
    if not isinstance(runs, list):
        return {"available": False, "status": "ERROR", "error": "Airflow response has no valid dag_runs list"}
    return {**result, "latest_run": runs[0] if runs else None, "message": "no run found" if not runs else None}


def trigger_training_dag(dag_id: str = "ml_training_pipeline", conf: dict[str, Any] | None = None) -> dict[str, Any]:
    if dag_id != "ml_training_pipeline":
        return {"available": False, "status": "REJECTED", "error": "DAG is not approved"}
    if conf is not None and len(conf) > 20:
        raise ValueError("configuration is too large")
    result = AirflowClient()._request("POST", f"/dags/{dag_id}/dagRuns", json={"conf": conf or {}})
    if result.get("available"):
        payload = result["response"]
        if not payload.get("dag_run_id") and not payload.get("run_id"):
            return {"available": False, "status": "ERROR", "error": "Airflow response omitted the DAG run ID"}
        result["dag_run_id"] = payload.get("dag_run_id") or payload.get("run_id")
    return result


def identify_failed_tasks(dag_id: str = "ml_training_pipeline", dag_run_id: str = "latest") -> dict[str, Any]:
    if not _ID.fullmatch(dag_id) or not _ID.fullmatch(dag_run_id):
        raise ValueError("invalid DAG identifier")
    result = AirflowClient()._request("GET", f"/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances")
    if not result.get("available"):
        return result
    tasks = result["response"].get("task_instances")
    if not isinstance(tasks, list):
        return {"available": False, "status": "ERROR", "error": "Airflow response has no valid task_instances list"}
    result["failed_tasks"] = [
        task for task in tasks
        if isinstance(task, dict) and task.get("state") in {"failed", "upstream_failed"}
    ]
    return result


def get_task_diagnostics(dag_id: str, dag_run_id: str, task_id: str, try_number: int = 1) -> dict[str, Any]:
    if not all(_ID.fullmatch(value) for value in (dag_id, dag_run_id, task_id)) or not 1 <= try_number <= 20:
        raise ValueError("invalid task diagnostic request")
    return AirflowClient()._request(
        "GET",
        f"/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{try_number}",
        allow_text=True,
    )

