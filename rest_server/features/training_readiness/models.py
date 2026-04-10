from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rest_server.features.intent_schema.models import IntentSchemaResult


class TrainingReadinessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_schema: IntentSchemaResult
    top_k: int = Field(default=5, ge=1, le=20)
    disable_llm: bool = True
    api_base: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout_seconds: int = Field(default=60, ge=1, le=300)
    cache_dir: str | None = None


class TrainingGateIssue(BaseModel):
    severity: str
    code: str
    message: str


class TrainingReadinessSummary(BaseModel):
    gate_status: str
    total_fields: int = 0
    direct_count: int = 0
    derivable_count: int = 0
    missing_count: int = 0
    selected_dataset_count: int = 0
    real_column_aware_dataset_count: int = 0


class TrainingReadinessResponse(BaseModel):
    status: str
    message: str
    summary: TrainingReadinessSummary
    issues: list[TrainingGateIssue] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    assembly_plan: dict = Field(default_factory=dict)
