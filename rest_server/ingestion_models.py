from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


JobStatus = Literal["Pending", "Crawling", "AI_Validating", "Completed", "Failed"]
ArtifactStatus = Literal["pending_review", "approved", "rejected", "failed"]


class IngestionFieldMapping(BaseModel):
    source_header: str
    target_field: str
    rationale: str


class AIValidationResult(BaseModel):
    pass_: bool = Field(alias="pass")
    confidence: float = Field(ge=0.0, le=1.0)
    resource_type: str | None = None
    field_mappings: list[IngestionFieldMapping] = Field(default_factory=list)
    summary: str
    license_guess: str | None = None
    recommended_title: str | None = None
    reject_reasons: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class DatasheetDraftModel(BaseModel):
    title: str
    description: str
    resource_type: str
    creators: list[str] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    publisher: str | None = None
    license_guess: str | None = None
    potential_uses: list[str] = Field(default_factory=list)
    related_download_url: str | None = None
    sample_notes: list[str] = Field(default_factory=list)


class ScrapeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl


class ScraperJobSummary(BaseModel):
    id: int
    source_url: str
    status: JobStatus
    requested_by: str | None = None
    page_title: str | None = None
    discovered_csv_count: int = 0
    processed_csv_count: int = 0
    passed_csv_count: int = 0
    created_artifact_count: int = 0
    error_message: str | None = None
    created_at: str
    updated_at: str
    completed_at: str | None = None


class ScraperJobDetail(ScraperJobSummary):
    discovered_csv_urls: list[str] = Field(default_factory=list)
    recent_failures: list[str] = Field(default_factory=list)


class IngestionArtifactSummary(BaseModel):
    id: int
    job_id: int
    source_url: str
    page_title: str | None = None
    csv_url: str
    status: ArtifactStatus
    title: str
    created_by: str | None = None
    reviewed_by: str | None = None
    review_notes: str | None = None
    created_at: str
    updated_at: str
    reviewed_at: str | None = None


class IngestionArtifactDetail(IngestionArtifactSummary):
    headers_sample: list[str] = Field(default_factory=list)
    rows_sample: list[dict[str, Any]] = Field(default_factory=list)
    validation_result: AIValidationResult
    datasheet_draft: DatasheetDraftModel
    staged_csv_download_url: str
    staged_schema_download_url: str
    staged_csv_path: str
    staged_schema_path: str


class ArtifactReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["approved", "rejected"]
    review_notes: str | None = Field(default=None, max_length=4000)

