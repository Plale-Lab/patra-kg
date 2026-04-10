from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rest_server.agent_tool_models import SearchCandidateModel
from rest_server.features.intent_schema.models import IntentSchemaResult


class MetadataDiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_schema: IntentSchemaResult
    top_k: int = Field(default=5, ge=1, le=20)
    disable_llm: bool = True
    api_base: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout_seconds: int = Field(default=60, ge=1, le=300)
    cache_dir: str | None = None


class MetadataDiscoveryCoverage(BaseModel):
    direct_count: int = 0
    derivable_count: int = 0
    missing_count: int = 0
    total_fields: int = 0


class MetadataDiscoveryResponse(BaseModel):
    status: str
    message: str
    query_schema: dict
    candidate_count: int = 0
    winner_dataset_id: str | None = None
    winner_coverage: MetadataDiscoveryCoverage | None = None
    pool_mode: str = "public"
    ranking: list[SearchCandidateModel] = Field(default_factory=list)


class InternalAssetIndexStatusResponse(BaseModel):
    status: str
    export_file: str | None = None
    export_exists: bool = False
    cache_file: str
    cache_exists: bool = False
    pair_count: int = 0
    counts_by_asset_type: dict[str, int] = Field(default_factory=dict)
    counts_by_source_family: dict[str, int] = Field(default_factory=dict)
    counts_by_quality: dict[str, int] = Field(default_factory=dict)
    generated_at: str | None = None
    export_last_modified: str | None = None


class InternalAssetIndexRefreshResponse(BaseModel):
    status: str
    message: str
    export_file: str
    cache_file: str
    pair_count: int = 0
    counts_by_asset_type: dict[str, int] = Field(default_factory=dict)
    counts_by_source_family: dict[str, int] = Field(default_factory=dict)
    counts_by_quality: dict[str, int] = Field(default_factory=dict)
    generated_at: str | None = None


class ColumnProfileResponse(BaseModel):
    name: str
    canonical_type: str
    json_schema_type: str
    inferred_format: str | None = None
    description: str
    sample_size: int
    non_null_count: int
    null_count: int
    null_ratio: float
    unique_count: int
    top_values: list[str] = Field(default_factory=list)
    enum_values: list[str | bool] | None = None
    min_value: float | str | None = None
    max_value: float | str | None = None
    nullable: bool = True


class UploadedCsvProfileResponse(BaseModel):
    status: str
    message: str
    source_label: str
    column_count: int = 0
    sample_row_count: int = 0
    columns: list[ColumnProfileResponse] = Field(default_factory=list)
    schema_contract: dict = Field(default_factory=dict)
