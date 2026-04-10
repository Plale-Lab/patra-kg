from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IntentSchemaStarter(BaseModel):
    title: str
    prompt: str


class IntentSchemaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_text: str = Field(min_length=8, max_length=6000)
    context: str | None = Field(default=None, max_length=4000)
    max_fields: int = Field(default=8, ge=3, le=20)


class IntentSchemaField(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    data_type: str = Field(min_length=1, max_length=64)
    semantic_role: Literal["target", "feature", "identifier", "timestamp", "grouping", "unknown"] = "unknown"
    description: str = Field(min_length=1, max_length=600)
    expected_range: str | None = Field(default=None, max_length=300)
    distribution_expectation: str | None = Field(default=None, max_length=300)
    required: bool = True
    notes: str | None = Field(default=None, max_length=300)


class IntentSchemaResult(BaseModel):
    intent_summary: str = Field(min_length=1, max_length=500)
    task_type: Literal["classification", "regression", "ranking", "forecasting", "unknown"] = "unknown"
    entity_grain: str = Field(min_length=1, max_length=300)
    target_column: str = Field(min_length=1, max_length=128)
    label_definition: str = Field(min_length=1, max_length=500)
    prediction_horizon: str | None = Field(default=None, max_length=200)
    ambiguity_warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    schema_fields: list[IntentSchemaField] = Field(default_factory=list)


class IntentSchemaResponse(IntentSchemaResult):
    mode: Literal["llm", "code_fallback"]
    provider: str
    model_used: str | None = None
    starter_prompts: list[IntentSchemaStarter] = Field(default_factory=list)


class IntentSchemaBootstrapResponse(BaseModel):
    enabled: bool
    provider: str
    starter_prompts: list[IntentSchemaStarter] = Field(default_factory=list)
