from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MvpDemoReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_text: str = Field(min_length=8, max_length=6000)
    context: str | None = Field(default=None, max_length=4000)
    max_fields: int = Field(default=8, ge=3, le=20)
    top_k: int = Field(default=5, ge=1, le=20)
    disable_llm: bool = True
    preview_row_limit: int = Field(default=5, ge=1, le=20)


class ExecutiveSummaryItem(BaseModel):
    label: str
    value: str
    tone: str = "neutral"


class MvpDemoReportResponse(BaseModel):
    status: str
    message: str
    executive_summary: list[ExecutiveSummaryItem] = Field(default_factory=list)
    schema_result: dict = Field(default_factory=dict)
    metadata_discovery: dict = Field(default_factory=dict)
    assembly_plan: dict = Field(default_factory=dict)
    composition_preview: dict = Field(default_factory=dict)
    training_stub: dict = Field(default_factory=dict)
