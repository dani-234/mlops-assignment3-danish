"""FastAPI inference and conversational agent interface."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from agent.orchestrator import MLOpsAgentOrchestrator
from agent.schemas import AgentQueryRequest, AgentQueryResponse, HealthResponse, PredictionRequest

app = FastAPI(title="Production MLOps Agent", version="1.0.0")
orchestrator = MLOpsAgentOrchestrator()


def _model() -> tuple[Any, dict[str, Any], Any]:
    import joblib
    path = Path(os.getenv("MODEL_ARTIFACT_PATH", "artifacts/model.joblib"))
    metadata_path = path.with_name("model_metadata.json")
    if not path.is_file():
        raise FileNotFoundError(f"model artifact is unavailable: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    preprocessor_path = Path(metadata.get("preprocessor_path", path.with_name("preprocessor.joblib")))
    preprocessor = joblib.load(preprocessor_path) if preprocessor_path.is_file() else None
    return joblib.load(path), metadata, preprocessor


@app.get("/health/live", response_model=HealthResponse)
def live() -> HealthResponse:
    return HealthResponse(status="ok", details={"service": "alive"})


@app.get("/health/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    try:
        _, metadata, _ = _model()
        return HealthResponse(status="ready", details={"features": metadata.get("features", [])})
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "error": str(exc)}) from exc


@app.get("/version")
def version() -> dict[str, Any]:
    try:
        _, metadata, _ = _model()
        return {"application_version": app.version, "model_version": os.getenv("MODEL_VERSION", "unknown"), "dataset_provenance": metadata.get("provenance", "unavailable")}
    except Exception as exc:
        return {"application_version": app.version, "model_version": "unavailable", "dataset_provenance": "unavailable", "error": str(exc)}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        model, metadata, preprocessor = _model()
        started = time.perf_counter()
        columns = metadata.get("features", ["feature_a", "feature_b", "feature_c", "feature_d"])
        features = pd.DataFrame([request.features], columns=columns)
        if preprocessor is not None:
            features = preprocessor["scaler"].transform(features)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max() if hasattr(model, "predict_proba") else None
        return {"prediction": int(prediction), "probability": probability, "model_version": os.getenv("MODEL_VERSION", "unknown"), "latency_ms": (time.perf_counter() - started) * 1000, "features": metadata.get("features", [])}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"prediction unavailable: {exc}") from exc


@app.post("/agent/query", response_model=AgentQueryResponse)
def query(request: AgentQueryRequest) -> AgentQueryResponse:
    return orchestrator.query(request.query, request.request_id)

