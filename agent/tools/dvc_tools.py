"""Safe DVC/Git provenance inspection tools."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

DATASET = "data/raw_dataset.csv"
_REVISION = re.compile(r"^[A-Za-z0-9._/-]{1,120}$")


def _repo(path: str) -> Path:
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError("repository path is not a directory")
    return root


def _run(args: list[str], cwd: Path, timeout: float = 15) -> subprocess.CompletedProcess[str]:
    allowed = {
        ("git", "status"),
        ("git", "rev-parse"),
        ("git", "show"),
        ("git", "worktree"),
        ("dvc", "status"),
        ("dvc", "remote"),
        ("dvc", "pull"),
        ("dvc", "checkout"),
    }
    is_module_dvc = (
        len(args) >= 4
        and args[0] == sys.executable
        and args[1:3] == ["-m", "dvc"]
        and args[3] in {"status", "remote", "pull", "checkout"}
    )
    if tuple(args[:2]) not in allowed and not is_module_dvc:
        raise ValueError("operation is not allowed")
    return subprocess.run(args, cwd=cwd, shell=False, timeout=timeout, check=False, capture_output=True, text=True)


def _dvc_command(*arguments: str) -> list[str]:
    executable = shutil.which("dvc")
    return [executable, *arguments] if executable else [sys.executable, "-m", "dvc", *arguments]


def _metadata(root: Path, dataset_path: str) -> dict[str, Any]:
    relative = Path(dataset_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("dataset path must remain inside the repository")
    target = root / relative
    dvc_file = Path(str(target) + ".dvc")
    if not dvc_file.is_file():
        return {"available": False, "reason": f"missing DVC metadata: {dvc_file.name}"}
    try:
        content = dvc_file.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        if not isinstance(parsed, dict):
            raise ValueError("DVC metadata is not a mapping")
        outs = parsed.get("outs")
        if not isinstance(outs, list) or not outs or not isinstance(outs[0], dict):
            raise ValueError("DVC metadata has no valid output entry")
        checksum = outs[0].get("md5")
        return {
            "available": True,
            "path": str(relative),
            "metadata_file": str(dvc_file.relative_to(root)),
            "checksum": checksum,
            "size": outs[0].get("size"),
            "raw": parsed,
        }
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return {"available": False, "reason": str(exc)}


def list_dvc_datasets(repo_path: str = ".") -> dict[str, Any]:
    root = _repo(repo_path)
    files = [str(path.relative_to(root)) for path in root.rglob("*.dvc") if ".git" not in path.parts]
    pipeline = root / "dvc.yaml"
    stages: list[str] = []
    if pipeline.is_file():
        try:
            parsed_pipeline = yaml.safe_load(pipeline.read_text(encoding="utf-8"))
            stage_map = parsed_pipeline.get("stages", {}) if isinstance(parsed_pipeline, dict) else {}
            stages = sorted(stage_map) if isinstance(stage_map, dict) else []
        except (OSError, yaml.YAMLError):
            stages = []
    return {"datasets": [{"metadata_file": name, "tracking": "DVC"} for name in sorted(files)], "pipeline_stages": stages, "repository": str(root)}


def get_current_dataset_version(dataset_path: str = DATASET, experiment_or_run_id: str | None = None) -> dict[str, Any]:
    root = _repo(".")
    metadata = _metadata(root, dataset_path)
    result: dict[str, Any] = {"dataset": dataset_path, "provenance": metadata}
    target = root / dataset_path
    if target.is_file():
        result["working_copy_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    commit = _run(["git", "rev-parse", "HEAD"], root)
    result["git_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
    result["recorded"] = bool(metadata.get("available") and metadata.get("checksum"))
    result["run_id"] = experiment_or_run_id
    try:
        remotes = _run(_dvc_command("remote", "list"), root)
        result["dvc_remotes"] = [
            line.split()[0]
            for line in remotes.stdout.splitlines()
            if line.split()
        ] if remotes.returncode == 0 else []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result["dvc_remotes"] = []
    if experiment_or_run_id:
        try:
            from mlflow.tracking import MlflowClient

            run = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")).get_run(experiment_or_run_id)
            result["mlflow_provenance"] = {
                key: run.data.tags.get(key)
                for key in ("dvc_dataset_version", "dataset_md5", "git_commit")
            }
        except Exception as exc:
            result["mlflow_provenance_error"] = f"{type(exc).__name__}: {exc}"
    return result


def compare_dataset_versions(rev_a: str, rev_b: str, dataset_path: str = DATASET) -> dict[str, Any]:
    if not _REVISION.fullmatch(rev_a) or not _REVISION.fullmatch(rev_b):
        raise ValueError("invalid Git revision")
    relative = Path(dataset_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("dataset path must remain inside the repository")
    root = _repo(".")
    values = {}
    for label, revision in (("a", rev_a), ("b", rev_b)):
        result = _run(["git", "show", f"{revision}:{relative}.dvc"], root)
        if result.returncode != 0:
            values[label] = None
            continue
        try:
            parsed = yaml.safe_load(result.stdout)
            output = parsed.get("outs", [{}])[0] if isinstance(parsed, dict) else {}
            values[label] = {"checksum": output.get("md5"), "size": output.get("size")}
        except (yaml.YAMLError, AttributeError, IndexError, TypeError):
            values[label] = None
    changed = None if values["a"] is None or values["b"] is None else values["a"] != values["b"]
    return {"dataset": dataset_path, "revisions": {"a": rev_a, "b": rev_b}, "changed": changed, "metadata": values}


def detect_data_changes(repo_path: str = ".") -> dict[str, Any]:
    root = _repo(repo_path)
    status = _run(["git", "status", "--porcelain"], root)
    lines = status.stdout.splitlines() if status.returncode == 0 else []
    try:
        dvc_status = _run(_dvc_command("status", "--quiet"), root)
        dvc_available = True
    except FileNotFoundError:
        dvc_status = None
        dvc_available = False
    data_changes = [line for line in lines if line.strip().endswith((".dvc", ".csv", "dvc.yaml"))]
    return {
        "data_changes": data_changes,
        "dvc_changed": bool(dvc_status and (dvc_status.returncode != 0 or dvc_status.stdout.strip())),
        "unrelated_changes": [line for line in lines if line not in data_changes],
        "dvc_available": dvc_available,
    }


def explain_dataset_reproduction(target_version_or_commit: str, dataset_path: str = DATASET, execute: bool = False, confirm_action: bool = False) -> dict[str, Any]:
    relative = Path(dataset_path)
    if not _REVISION.fullmatch(target_version_or_commit) or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid revision or dataset path")
    commands = [
        ["git", "worktree", "add", "--detach", "<temporary-path>", target_version_or_commit],
        ["dvc", "pull", str(relative)],
        ["dvc", "checkout", str(relative)],
    ]
    if execute and not confirm_action:
        return {"status": "REJECTED", "reason": "explicit confirmation is required", "commands": commands}
    if not execute:
        return {"status": "EXPLAINED", "target": target_version_or_commit, "commands": commands}
    root = _repo(".")
    if not (root / ".git").exists():
        return {"status": "ERROR", "target": target_version_or_commit, "commands": commands, "error": "execution requires a Git repository"}
    with tempfile.TemporaryDirectory(prefix="dvc-reproduction-") as temporary_path:
        worktree = Path(temporary_path) / "worktree"
        created = _run(
            ["git", "worktree", "add", "--detach", str(worktree), target_version_or_commit],
            root,
        )
        if created.returncode != 0:
            return {"status": "ERROR", "target": target_version_or_commit, "commands": commands, "error": created.stderr.strip()}
        try:
            pull = _run(_dvc_command("pull", str(relative)), worktree)
            checkout = _run(_dvc_command("checkout", str(relative)), worktree)
            status = "EXECUTED" if pull.returncode == 0 and checkout.returncode == 0 else "ERROR"
            return {
                "status": status,
                "target": target_version_or_commit,
                "commands": commands,
                "worktree": str(worktree),
                "stdout": (pull.stdout + checkout.stdout).strip(),
                "error": (pull.stderr + checkout.stderr).strip() or None,
            }
        finally:
            _run(["git", "worktree", "remove", "--force", str(worktree)], root)

