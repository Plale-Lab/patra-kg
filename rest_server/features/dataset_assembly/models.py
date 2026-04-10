from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rest_server.features.intent_schema.models import IntentSchemaResult


class DatasetAssemblyPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_schema: IntentSchemaResult
    top_k: int = Field(default=5, ge=1, le=20)
    disable_llm: bool = True
    api_base: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout_seconds: int = Field(default=60, ge=1, le=300)
    cache_dir: str | None = None


class AssemblyPlanSummary(BaseModel):
    direct_count: int = 0
    derivable_count: int = 0
    missing_count: int = 0
    total_fields: int = 0
    selected_dataset_count: int = 0
    requires_manual_join_confirmation: bool = False


class FieldSourceResolution(BaseModel):
    target_field: str
    semantic_role: str
    resolution_status: str
    source_tier: str | None = None
    source_quality: str | None = None
    source_approved: bool | None = None
    source_dataset_id: str | None = None
    source_dataset_title: str | None = None
    source_family: str | None = None
    source_url: str | None = None
    source_column: str | None = None
    source_columns: list[str] = Field(default_factory=list)
    rationale: str
    notes: list[str] = Field(default_factory=list)


class SelectedDatasetSummary(BaseModel):
    dataset_id: str
    title: str
    source_tier: str
    source_quality: str | None = None
    approved: bool = False
    source_family: str
    source_url: str
    direct_fields: list[str] = Field(default_factory=list)
    derivable_fields: list[str] = Field(default_factory=list)


class JoinRequirement(BaseModel):
    level: str
    message: str
    datasets: list[str] = Field(default_factory=list)


class DatasetAssemblyPlanResponse(BaseModel):
    status: str
    message: str
    query_schema: dict = Field(default_factory=dict)
    pool_mode: str = "public"
    summary: AssemblyPlanSummary
    selected_datasets: list[SelectedDatasetSummary] = Field(default_factory=list)
    field_resolutions: list[FieldSourceResolution] = Field(default_factory=list)
    join_requirements: list[JoinRequirement] = Field(default_factory=list)
    fallback_recommendations: list[str] = Field(default_factory=list)


class DatasetCompositionPreviewRequest(BaseModel):
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


class CompositionManifestField(BaseModel):
    target_field: str
    resolution_status: str
    source_tier: str | None = None
    source_dataset_id: str | None = None
    source_columns: list[str] = Field(default_factory=list)
    included_in_preview: bool = False
    notes: list[str] = Field(default_factory=list)


class CompositionManifest(BaseModel):
    selected_dataset_ids: list[str] = Field(default_factory=list)
    selected_dataset_count: int = 0
    preview_mode: str
    blocked: bool = False
    block_reasons: list[str] = Field(default_factory=list)
    fields: list[CompositionManifestField] = Field(default_factory=list)


class DatasetCompositionPreviewResponse(BaseModel):
    status: str
    message: str
    assembly_plan: dict = Field(default_factory=dict)
    manifest: CompositionManifest
    preview_rows: list[dict] = Field(default_factory=list)
