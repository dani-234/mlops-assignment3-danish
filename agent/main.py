"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.failure_simulator import simulate_kubernetes_failure, simulate_quality_gate_failure
from agent.orchestrator import MLOpsAgentOrchestrator
from agent.schemas import AgentQueryRequest


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline or configured MLOps agent")
    parser.add_argument("query", nargs="?", help="natural-language query")
    parser.add_argument("--failure", choices=("quality-gate", "kubernetes"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = simulate_quality_gate_failure() if args.failure == "quality-gate" else simulate_kubernetes_failure() if args.failure == "kubernetes" else MLOpsAgentOrchestrator().query(args.query or input("Query: "), AgentQueryRequest(query=args.query or "interactive").request_id).model_dump(mode="json")
    rendered = json.dumps(value, indent=2)
    print(rendered)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "evidence.json").write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

