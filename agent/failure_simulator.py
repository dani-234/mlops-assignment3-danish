"""Deterministic, isolated failure-investigation fixtures."""

from __future__ import annotations


def simulate_quality_gate_failure() -> dict:
    score, threshold = 0.62, 0.80
    return {"simulated": True, "failure": "quality_gate", "status": "failed", "metric": "f1_score", "value": score, "threshold": threshold, "diagnosis": "Recorded F1 is below the required threshold.", "recommended_action": "Inspect data quality and model changes, then retrain only after correcting the cause."}


def simulate_kubernetes_failure() -> dict:
    return {"simulated": True, "failure": "kubernetes", "pod": {"name": "simulated-inference", "phase": "Running", "reason": "CrashLoopBackOff", "restarts": 6}, "evidence": ["container restart count is 6", "last termination reason is CrashLoopBackOff"], "diagnosis": "The simulated inference container is repeatedly restarting.", "recommended_action": "Inspect the actual pod logs and image/model artifact configuration."}

