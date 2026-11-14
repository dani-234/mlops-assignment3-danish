"""Validated contracts shared by the API, agent, and operational tools."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ToolResult(StrictModel):
    tool_name: str = Field(min_length=1, max_length=100)
    status: Literal["SUCCESS", "ERROR", "TIMEOUT", "REJECTED"]
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float = Field(ge=0)
    request_id: UUID

    @field_validator("error")
    @classmethod
    def empty_error_is_none(cls, value: str | None) -> str | None:
        return value or None


class AgentQueryRequest(StrictModel):
    query: str = Field(min_length=1, max_length=4000)
    request_id: UUID = Field(default_factory=uuid4)
    confirm_cluster_change: bool = False
    confirmation_token: str | None = Field(default=None, min_length=16, max_length=256)


class AgentQueryResponse(StrictModel):
    request_id: UUID
    answer: str
    known_facts: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    tool_traces: list[ToolResult] = Field(default_factory=list)
    confirmation_required: bool = False
    confirmation_token: str | None = None


class PredictionRequest(StrictModel):
    features: list[float] = Field(min_length=4, max_length=4)

    @field_validator("features")
    @classmethod
    def finite_features(cls, value: list[float]) -> list[float]:
        if any(not isinstance(item, (int, float)) or not (-1e12 < item < 1e12) for item in value):
            raise ValueError("features must contain finite numeric values")
        return [float(item) for item in value]


class HealthResponse(StrictModel):
    status: Literal["ok", "ready", "not_ready"]
    details: dict[str, Any] = Field(default_factory=dict)

