"""Reusable training and evaluation logic."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training.generate_dataset import FEATURES, TARGET


def _provenance(dataset_path: str) -> dict[str, str | None]:
    path = Path(dataset_path)
    digest = hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else None
    dvc_path = Path(f"{dataset_path}.dvc")
    dvc_checksum = None
    if dvc_path.is_file():
        for line in dvc_path.read_text(encoding="utf-8").splitlines():
            if "md5:" in line:
                dvc_checksum = line.split("md5:", 1)[1].strip()
                break
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            shell=False,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        git_commit = commit.stdout.strip() if commit.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        git_commit = None
    return {"dataset_md5": dvc_checksum or digest, "git_commit": git_commit}


def log_mlflow_run(result: dict[str, Any], dataset_path: str) -> dict[str, Any]:
    """Log the completed run when MLflow is installed, without hiding failures."""
    try:
        import mlflow

        provenance = _provenance(dataset_path)
        mlflow.set_experiment("mlops-agent-experiment")
        with mlflow.start_run(run_name="mlops-agent-training") as run:
            mlflow.log_params({"n_estimators": 100, "random_seed": result["seed"]})
            mlflow.log_metrics(result["metrics"])
            mlflow.set_tags(
                {
                    "dvc_dataset_version": provenance["dataset_md5"] or "unavailable",
                    "dataset_md5": provenance["dataset_md5"] or "unavailable",
                    "git_commit": provenance["git_commit"] or "unavailable",
                    "quality_gate": str(result["quality_gate"]).lower(),
                }
            )
            mlflow.log_artifact(result["model_path"], artifact_path="model")
            mlflow.log_artifact(result["preprocessor_path"], artifact_path="preprocessing")
            mlflow.log_artifact(result["metadata_path"], artifact_path="metadata")
            return {"logged": True, "run_id": run.info.run_id, "provenance": provenance}
    except Exception as exc:
        return {"logged": False, "error": f"{type(exc).__name__}: {exc}"}


def prepare_preprocessing_artifacts(
    dataset_path: str = "data/raw_dataset.csv",
    artifact_dir: str = "artifacts",
    seed: int = 42,
) -> dict[str, Any]:
    frame = pd.read_csv(dataset_path)
    if TARGET not in frame or any(feature not in frame for feature in FEATURES):
        raise ValueError("dataset is missing required features or target")
    if frame[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("dataset contains missing values")
    x_train, x_valid, y_train, y_valid = train_test_split(
        frame[FEATURES],
        frame[TARGET],
        test_size=0.2,
        random_state=seed,
        stratify=frame[TARGET],
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    preprocessor_path = root / "preprocessor.joblib"
    processed_path = root / "processed_data.joblib"
    joblib.dump({"scaler": scaler, "features": FEATURES}, preprocessor_path)
    joblib.dump(
        {
            "x_train": x_train_scaled,
            "x_valid": x_valid_scaled,
            "y_train": y_train.to_numpy(),
            "y_valid": y_valid.to_numpy(),
            "features": FEATURES,
        },
        processed_path,
    )
    return {
        "preprocessor_path": str(preprocessor_path),
        "processed_data_path": str(processed_path),
        "train_rows": len(x_train),
        "validation_rows": len(x_valid),
        "seed": seed,
    }


def train_and_evaluate(
    dataset_path: str = "data/raw_dataset.csv",
    artifact_dir: str = "artifacts",
    seed: int = 42,
    log_to_mlflow: bool = False,
) -> dict[str, Any]:
    preprocessing = prepare_preprocessing_artifacts(dataset_path, artifact_dir, seed)
    processed = joblib.load(preprocessing["processed_data_path"])
    model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    model.fit(processed["x_train"], processed["y_train"])
    predictions = model.predict(processed["x_valid"])
    probabilities = model.predict_proba(processed["x_valid"])[:, 1]
    metrics = {
        "f1_score": f1_score(processed["y_valid"], predictions),
        "accuracy": accuracy_score(processed["y_valid"], predictions),
        "precision": precision_score(processed["y_valid"], predictions, zero_division=0),
        "recall": recall_score(processed["y_valid"], predictions, zero_division=0),
        "roc_auc": roc_auc_score(processed["y_valid"], probabilities),
    }
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "model.joblib"
    metadata_path = root / "model_metadata.json"
    joblib.dump(model, model_path)
    provenance = _provenance(dataset_path)
    quality_gate = metrics["f1_score"] >= 0.80 and metrics["accuracy"] >= 0.80 and provenance["dataset_md5"] is not None
    result = {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        **preprocessing,
        "metrics": metrics,
        "seed": seed,
        "quality_gate": quality_gate,
        "provenance": provenance,
    }
    metadata_path.write_text(
        json.dumps({"features": FEATURES, "target": TARGET, **result}, indent=2),
        encoding="utf-8",
    )
    if log_to_mlflow:
        result["mlflow"] = log_mlflow_run(result, dataset_path)
    return result


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))

