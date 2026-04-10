from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rest_server.features.intent_schema.models import IntentSchemaResult


class BaselineTrainingStubRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_schema: IntentSchemaResult
    top_k: int = Field(default=5, ge=1, le=20)
    disable_llm: bool = True
    api_base: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout_seconds: int = Field(default=60, ge=1, le=300)
    cache_dir: str | None = None
    preview_row_limit: int = Field(default=5, ge=1, le=20)


class BaselineTrainingRecommendation(BaseModel):
    model_family: str
    reason: str
    training_scope: str


class DemoMetric(BaseModel):
    name: str
    value: str
    description: str
    simulated: bool = True


class EvalReportSection(BaseModel):
    title: str
    bullets: list[str] = Field(default_factory=list)


class BaselineTrainingStubSummary(BaseModel):
    execution_status: str
    gate_status: str
    task_type: str
    selected_dataset_count: int = 0
    preview_row_count: int = 0
    direct_count: int = 0
    derivable_count: int = 0
    missing_count: int = 0


class BaselineTrainingStubResponse(BaseModel):
    status: str
    message: str
    summary: BaselineTrainingStubSummary
    recommendation: BaselineTrainingRecommendation
    metrics: list[DemoMetric] = Field(default_factory=list)
    eval_report: list[EvalReportSection] = Field(default_factory=list)
    training_readiness: dict = Field(default_factory=dict)
    composition_preview: dict = Field(default_factory=dict)
