"""Read-only Kubernetes diagnostics and explicitly authorized mutations."""

from __future__ import annotations

import os
import secrets
import time
from typing import Any


class ConfirmationStore:
    def __init__(self) -> None:
        self._tokens: dict[str, tuple[float, str, str]] = {}

    def issue(self, operation: str, namespace: str, ttl: int = 120) -> str:
        token = secrets.token_urlsafe(24)
        self._tokens[token] = (time.time() + ttl, operation, namespace)
        return token

    def consume(self, token: str, operation: str, namespace: str) -> bool:
        item = self._tokens.pop(token, None)
        return bool(item and item[0] >= time.time() and item[1] == operation and item[2] == namespace)


confirmations = ConfirmationStore()


def _approved(namespace: str) -> bool:
    configured_namespace = os.getenv("APPROVED_NAMESPACE")
    return bool(configured_namespace) and namespace == configured_namespace


def _apis() -> tuple[Any, Any]:
    from kubernetes import client, config
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.CoreV1Api(), client.AppsV1Api()


def _error(exc: Exception) -> dict[str, Any]:
    return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _rollout_status(apps: Any, deployment_name: str, namespace: str) -> dict[str, Any]:
    deadline = time.time() + 30
    while time.time() < deadline:
        deployment = apps.read_namespaced_deployment(deployment_name, namespace)
        desired = deployment.spec.replicas
        ready = deployment.status.ready_replicas or 0
        available = deployment.status.available_replicas or 0
        if isinstance(desired, int) and isinstance(ready, int) and ready >= desired:
            return {"confirmed": True, "desired": desired, "ready": ready, "available": available}
        if not isinstance(desired, int) or not isinstance(ready, int):
            return {"confirmed": False, "error": "rollout status response was malformed"}
        time.sleep(1)
    return {"confirmed": False, "error": "rollout did not become ready within 30 seconds"}


def list_pods_and_deployments(namespace: str = "mlops-prod") -> dict[str, Any]:
    try:
        core, apps = _apis()
        pods = core.list_namespaced_pod(namespace).items
        deployments = apps.list_namespaced_deployment(namespace).items
        return {"available": True, "namespace": namespace, "pods": [{"name": p.metadata.name, "phase": p.status.phase, "restarts": sum((c.restart_count or 0) for c in (p.status.container_statuses or []))} for p in pods], "deployments": [{"name": d.metadata.name, "desired": d.spec.replicas, "ready": d.status.ready_replicas or 0, "available": d.status.available_replicas or 0} for d in deployments]}
    except Exception as exc:
        return _error(exc)


def get_pod_status_and_logs(pod_name: str, namespace: str = "mlops-prod", tail_lines: int = 50) -> dict[str, Any]:
    if not 1 <= tail_lines <= 500 or not pod_name.replace("-", "").replace(".", "").isalnum():
        raise ValueError("invalid pod request")
    try:
        core, _ = _apis()
        pod = core.read_namespaced_pod(pod_name, namespace)
        logs = None
        log_error = None
        try:
            logs = core.read_namespaced_pod_log(pod_name, namespace, tail_lines=tail_lines)
        except Exception as exc:
            log_error = f"{type(exc).__name__}: {exc}"
        return {
            "available": True,
            "pod": {
                "name": pod.metadata.name,
                "phase": pod.status.phase,
                "logs": logs,
                "log_error": log_error,
                "containers": [
                    {
                        "name": c.name,
                        "ready": c.ready,
                        "restarts": c.restart_count,
                        "state": str(c.state),
                        "last_state": str(c.last_state),
                    }
                    for c in (pod.status.container_statuses or [])
                ],
            },
        }
    except Exception as exc:
        return _error(exc)


def inspect_services(namespace: str = "mlops-prod") -> dict[str, Any]:
    try:
        core, _ = _apis()
        return {"available": True, "namespace": namespace, "services": [{"name": s.metadata.name, "type": s.spec.type, "cluster_ip": s.spec.cluster_ip, "ports": [{"port": p.port, "target_port": str(p.target_port)} for p in (s.spec.ports or [])], "selector": s.spec.selector or {}} for s in core.list_namespaced_service(namespace).items]}
    except Exception as exc:
        return _error(exc)


def diagnose_unhealthy_pods(namespace: str = "mlops-prod") -> dict[str, Any]:
    try:
        core, apps = _apis()
        pods = core.list_namespaced_pod(namespace).items
        deployments = apps.list_namespaced_deployment(namespace).items
        findings: list[dict[str, Any]] = []
        for pod in pods:
            statuses = pod.status.container_statuses or []
            reasons = [
                state.reason
                for status in statuses
                for state in (status.state, status.last_state)
                if state is not None and getattr(state, "reason", None)
            ]
            restarts = sum(status.restart_count or 0 for status in statuses)
            not_ready = [status.name for status in statuses if not status.ready]
            events = core.list_namespaced_event(
                namespace,
                field_selector=f"involvedObject.name={pod.metadata.name}",
            ).items
            warnings = [
                {"reason": event.reason, "message": event.message, "type": event.type}
                for event in events
                if event.type == "Warning"
            ]
            evidence = {
                "phase": pod.status.phase,
                "reasons": reasons,
                "restart_count": restarts,
                "not_ready_containers": not_ready,
                "warning_events": warnings,
            }
            if pod.status.phase != "Running" or reasons or restarts > 3 or not_ready or warnings:
                findings.append(
                    {
                        "pod": pod.metadata.name,
                        "severity": "confirmed",
                        "evidence": evidence,
                        "causes": reasons or (["repeated restarts"] if restarts > 3 else ["unsuccessful readiness"]),
                    }
                )
        return {
            "available": True,
            "namespace": namespace,
            "healthy": not findings,
            "findings": findings,
            "deployments": [
                {
                    "name": item.metadata.name,
                    "desired": item.spec.replicas,
                    "ready": item.status.ready_replicas or 0,
                    "available": item.status.available_replicas or 0,
                }
                for item in deployments
            ],
        }
    except Exception as exc:
        return _error(exc)


def get_deployed_app_version(deployment_name: str = "mlops-inference-deployment", namespace: str = "mlops-prod") -> dict[str, Any]:
    try:
        core, apps = _apis()
        deployment = apps.read_namespaced_deployment(deployment_name, namespace)
        containers = deployment.spec.template.spec.containers
        config_map = None
        config_map_error = None
        try:
            config_map = core.read_namespaced_config_map("mlops-config", namespace)
        except Exception as exc:
            config_map_error = f"{type(exc).__name__}: {exc}"
        return {
            "available": True,
            "deployment": deployment_name,
            "images": [item.image for item in containers],
            "labels": deployment.metadata.labels or {},
            "model_metadata": config_map.data if config_map is not None and config_map.data else {},
            "model_metadata_error": config_map_error,
        }
    except Exception as exc:
        return _error(exc)


def scale_or_restart_deployment(deployment_name: str, namespace: str = "mlops-prod", replicas: int = 2, confirm_action: bool = False, confirmation_token: str | None = None, operation: str = "scale") -> dict[str, Any]:
    if operation not in {"scale", "restart"} or not 1 <= replicas <= 10 or not _approved(namespace):
        return {"status": "REJECTED", "error": "operation, replica count, or approved namespace is invalid"}
    if not confirm_action:
        return {"status": "REJECTED", "confirmation_required": True, "confirmation_token": confirmations.issue(operation, namespace), "error": "explicit confirmation is required"}
    if not confirmation_token or not confirmations.consume(confirmation_token, operation, namespace):
        return {"status": "REJECTED", "error": "invalid, expired, or reused confirmation token"}
    try:
        _, apps = _apis()
        if operation == "scale":
            body = {"spec": {"replicas": replicas}}
            observed = apps.patch_namespaced_deployment_scale(deployment_name, namespace, body)
        else:
            body = {"spec": {"template": {"metadata": {"annotations": {"mlops-agent/restarted-at": str(time.time())}}}}}
            observed = apps.patch_namespaced_deployment(deployment_name, namespace, body)
        return {
            "status": "SUCCESS",
            "operation": operation,
            "observed_replicas": getattr(getattr(observed, "spec", None), "replicas", None),
            "rollout": _rollout_status(apps, deployment_name, namespace),
        }
    except Exception as exc:
        return {"status": "ERROR", **_error(exc)}

