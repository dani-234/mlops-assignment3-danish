"""Deterministic and provider-independent MLOps request orchestration."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable
from uuid import UUID

from agent.logger import logger
from agent.schemas import AgentQueryResponse, ToolResult
from agent.tools import airflow_tools, dvc_tools, kubernetes_tools, mlflow_tools


class MLOpsAgentOrchestrator:
    def __init__(self, tools: dict[str, Callable[..., Any]] | None = None) -> None:
        self.tools = tools or {
            "best_model": mlflow_tools.find_best_run,
            "run_details": mlflow_tools.get_run_details,
            "dataset_version": dvc_tools.get_current_dataset_version,
            "data_changes": dvc_tools.detect_data_changes,
            "deployment_version": kubernetes_tools.get_deployed_app_version,
            "pod_diagnosis": kubernetes_tools.diagnose_unhealthy_pods,
            "dag_status": airflow_tools.get_dag_run_status,
            "failed_tasks": airflow_tools.identify_failed_tasks,
        }

    def _invoke(self, name: str, request_id: UUID, **kwargs: Any) -> tuple[ToolResult, Any]:
        started = time.perf_counter()
        try:
            data = self.tools[name](**kwargs)
            status = "SUCCESS" if data.get("available", True) else "ERROR"
            error = data.get("error") if isinstance(data, dict) else None
        except TimeoutError as exc:
            data, status, error = {}, "TIMEOUT", str(exc)
        except (ValueError, TypeError) as exc:
            data, status, error = {}, "REJECTED", str(exc)
        except Exception as exc:
            data, status, error = {}, "ERROR", f"{type(exc).__name__}: {exc}"
        duration = (time.perf_counter() - started) * 1000
        trace = ToolResult(tool_name=name, status=status, data=data if isinstance(data, dict) else {"value": data}, error=error, duration_ms=duration, request_id=request_id)
        logger.tool_event(request_id=str(request_id), tool_name=name, status=status, duration_ms=duration, details=data)
        return trace, data

    def _openai_query(self, query: str, request_id: UUID) -> AgentQueryResponse:
        from openai import OpenAI

        definitions = [
            {"type": "function", "function": {"name": "find_best_run", "description": "Find the best recorded MLflow run.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
            {"type": "function", "function": {"name": "get_current_dataset_version", "description": "Inspect recorded DVC dataset provenance.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
            {"type": "function", "function": {"name": "get_deployed_app_version", "description": "Inspect the deployed application version.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
            {"type": "function", "function": {"name": "diagnose_unhealthy_pods", "description": "Diagnose observed Kubernetes pod failures.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
        ]
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Use only the supplied tools. Never invent operational facts. State uncertainty when a tool is unavailable."},
            {"role": "user", "content": query},
        ]
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            tools=definitions,
            tool_choice="auto",
            timeout=20,
        )
        message = response.choices[0].message
        traces: list[ToolResult] = []
        evidence: list[dict[str, Any]] = []
        dispatch = {
            "find_best_run": ("best_model", {"experiment_name": "mlops-agent-experiment", "metric": "f1_score", "top_k": 5}),
            "get_current_dataset_version": ("dataset_version", {}),
            "get_deployed_app_version": ("deployment_version", {}),
            "diagnose_unhealthy_pods": ("pod_diagnosis", {}),
        }
        for call in message.tool_calls or []:
            if call.function.name not in dispatch:
                traces.append(ToolResult(tool_name=call.function.name, status="REJECTED", data={}, error="tool is not approved", duration_ms=0, request_id=request_id))
                continue
            name, kwargs = dispatch[call.function.name]
            try:
                parsed = json.loads(call.function.arguments or "{}")
                if parsed:
                    raise ValueError("this tool accepts no LLM-controlled arguments")
            except (json.JSONDecodeError, ValueError) as exc:
                traces.append(ToolResult(tool_name=name, status="REJECTED", data={}, error=str(exc), duration_ms=0, request_id=request_id))
                continue
            trace, data = self._invoke(name, request_id, **kwargs)
            traces.append(trace)
            evidence.append({"tool": name, "data": data})
        if not traces:
            return AgentQueryResponse(request_id=request_id, answer=message.content or "The provider returned no operational evidence.")
        return AgentQueryResponse(
            request_id=request_id,
            answer=message.content or "The provider requested approved operational tools.",
            known_facts=[f"{item['tool']} returned evidence." for item in evidence if item["data"]],
            uncertainties=[trace.error or f"{trace.tool_name} returned no evidence." for trace in traces if trace.status != "SUCCESS"],
            recommended_actions=["Review each recorded tool result before taking operational action."],
            tool_traces=traces,
        )

    def _offline_query(self, query: str, request_id: UUID) -> AgentQueryResponse:
        normalized = query.lower()
        traces: list[ToolResult] = []
        facts: list[str] = []
        uncertainties: list[str] = []
        recommendations: list[str] = []
        if any(word in normalized for word in ("best model", "recent model", "retraining", "deployment status")):
            sequence = [
                ("best_model", {"experiment_name": "mlops-agent-experiment", "metric": "f1_score", "top_k": 5}),
            ]
            selected_run = None
            for name, kwargs in sequence:
                trace, data = self._invoke(name, request_id, **kwargs)
                traces.append(trace)
                if name == "best_model":
                    selected_run = data.get("best_run", {}).get("run_id") if isinstance(data, dict) and isinstance(data.get("best_run"), dict) else None
            dataset_kwargs = {"experiment_or_run_id": selected_run} if selected_run else {}
            for name, kwargs in (
                ("dataset_version", dataset_kwargs),
                ("data_changes", {}),
                ("deployment_version", {}),
                ("pod_diagnosis", {}),
                ("dag_status", {}),
                ("failed_tasks", {}),
            ):
                trace, data = self._invoke(name, request_id, **kwargs)
                traces.append(trace)
                if trace.status != "SUCCESS":
                    uncertainties.append(f"{name} was unavailable: {trace.error or 'no evidence returned'}")
            if selected_run:
                trace, data = self._invoke("run_details", request_id, run_id=selected_run)
                traces.append(trace)
                provenance = data.get("provenance", {}) if isinstance(data, dict) else {}
                if provenance:
                    facts.append(f"Selected run {selected_run} recorded provenance: {provenance}.")
            else:
                uncertainties.append("No eligible MLflow run was available, so model selection is unknown.")
            for trace in traces:
                if trace.status == "SUCCESS" and trace.data:
                    facts.append(f"{trace.tool_name} returned observed data.")
            recommendations.append("Retraining is not justified by a dataset change alone; require measured degradation, a failed quality gate, or a freshness breach.")
            answer = "Multi-tool assessment completed from the evidence that was available."
        elif "dataset" in normalized or "provenance" in normalized:
            trace, _ = self._invoke("dataset_version", request_id)
            traces.append(trace)
            answer = "Dataset provenance inspection completed." if trace.status == "SUCCESS" else "Dataset provenance could not be established."
            if trace.status != "SUCCESS":
                uncertainties.append(trace.error or "DVC unavailable")
        else:
            answer = "Offline planner supports dataset provenance and multi-tool model/deployment assessment queries."
            recommendations.append("Ask for the best recent model, dataset provenance, deployment status, and retraining assessment.")
        return AgentQueryResponse(request_id=request_id, answer=answer, known_facts=facts, uncertainties=uncertainties, recommended_actions=recommendations, tool_traces=traces)

    def query(self, query: str, request_id: UUID) -> AgentQueryResponse:
        provider = os.getenv("AGENT_LLM_PROVIDER", "offline").lower()
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            try:
                return self._openai_query(query, request_id)
            except Exception as exc:
                fallback = self._offline_query(query, request_id)
                fallback.uncertainties.insert(0, f"OpenAI provider failed; deterministic fallback used: {type(exc).__name__}: {exc}")
                return fallback
        if provider not in {"offline", ""}:
            fallback = self._offline_query(query, request_id)
            fallback.uncertainties.insert(0, f"Provider {provider!r} is unavailable; deterministic fallback used.")
            return fallback
        return self._offline_query(query, request_id)

