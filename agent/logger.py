"""Structured logging with conservative secret redaction."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

REDACTED = "***REDACTED***"
SECRET_KEYS = {"api_key", "token", "password", "authorization", "kubeconfig", "secret"}
SECRET_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(https?://)([^/@\s]+):([^/@\s]+)@"),
    re.compile(r"(?i)((?:api[_-]?key|token|password|secret|authorization)\s*[=:]\s*)[^\s,;]+"),
]


def redact(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {name: REDACTED if name.lower() in SECRET_KEYS else redact(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [redact(item, key) for item in value]
    if isinstance(value, str):
        result = value
        for pattern in SECRET_PATTERNS:
            result = pattern.sub(lambda match: match.group(1) + REDACTED, result)
        return result
    return value


class JsonLogger:
    def __init__(self, name: str = "mlops-agent") -> None:
        self._logger = logging.getLogger(name)

    def tool_event(self, *, request_id: str, tool_name: str, status: str, duration_ms: float, details: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "tool_name": tool_name,
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "details": redact(details),
        }
        self._logger.info(json.dumps(redact(record), sort_keys=True))


logger = JsonLogger()

