"""Airflow 3 training DAG; imports remain usable for offline structure tests."""

from datetime import timedelta

try:
    from airflow.providers.standard.operators.python import PythonOperator
    from airflow.sdk import DAG
except ImportError:
    DAG = None
    PythonOperator = None


def validate_dataset() -> dict:
    import hashlib
    from pathlib import Path

    import pandas as pd

    from training.train import FEATURES, TARGET
    frame = pd.read_csv("data/raw_dataset.csv")
    dvc_file = Path("data/raw_dataset.csv.dvc")
    if (
        len(frame) < 100
        or any(feature not in frame or frame[feature].isna().any() for feature in FEATURES)
        or TARGET not in frame
        or not dvc_file.is_file()
    ):
        raise ValueError("dataset validation failed")
    checksum = next(
        (line.split("md5:", 1)[1].strip() for line in dvc_file.read_text(encoding="utf-8").splitlines() if "md5:" in line),
        None,
    )
    actual = hashlib.md5(Path("data/raw_dataset.csv").read_bytes()).hexdigest()
    if not checksum or checksum != actual:
        raise ValueError("dataset validation failed: DVC checksum does not match the dataset")
    return {"rows": len(frame), "features": FEATURES}


def preprocess_data(**_: object) -> dict:
    from training.train import prepare_preprocessing_artifacts

    return prepare_preprocessing_artifacts(
        dataset_path="data/raw_dataset.csv",
        artifact_dir="artifacts",
        seed=42,
    )


def train_model(**_: object) -> dict:
    from training.train import train_and_evaluate

    return train_and_evaluate()


def evaluate_and_log_mlflow(**_: object) -> dict:
    import json
    from pathlib import Path

    from training.train import log_mlflow_run

    result = json.loads(Path("artifacts/model_metadata.json").read_text(encoding="utf-8"))
    result["mlflow"] = log_mlflow_run(result, "data/raw_dataset.csv")
    Path("artifacts/model_metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def apply_model_quality_gate(**_: object) -> None:
    import json
    from pathlib import Path
    metadata = json.loads(Path("artifacts/model_metadata.json").read_text(encoding="utf-8"))
    metrics = metadata.get("metrics", {})
    provenance = metadata.get("provenance", {})
    if metrics.get("f1_score", 0) < 0.80 or metrics.get("accuracy", 0) < 0.80:
        raise ValueError("quality gate failed: f1_score and accuracy must be at least 0.80")
    if not provenance.get("dataset_md5") or not Path("data/raw_dataset.csv.dvc").is_file():
        raise ValueError("quality gate failed: DVC provenance is unavailable")


if DAG is not None:
    with DAG("ml_training_pipeline", schedule=None, catchup=False, default_args={"retries": 2, "retry_delay": timedelta(minutes=1)}) as dag:
        validate = PythonOperator(task_id="validate_dataset", python_callable=validate_dataset)
        preprocess = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
        train = PythonOperator(task_id="train_model", python_callable=train_model)
        evaluate = PythonOperator(task_id="evaluate_and_log_mlflow", python_callable=evaluate_and_log_mlflow)
        gate = PythonOperator(task_id="apply_model_quality_gate", python_callable=apply_model_quality_gate)
        validate >> preprocess >> train >> evaluate >> gate

