from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SubmissionType = Literal["model_card", "datasheet"]
SubmissionStatus = Literal["pending", "in_progress", "approved", "rejected"]
TicketStatus = Literal["open", "in_progress", "resolved"]


class TicketCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submitted_by: str = Field(min_length=1, max_length=255)
    subject: str = Field(min_length=1, max_length=255)
    category: str = Field(default="General", min_length=1, max_length=100)
    priority: str = Field(default="Medium", min_length=1, max_length=50)
    description: str = Field(min_length=1)


class TicketUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: TicketStatus
    admin_response: str | None = None


class TicketRecord(BaseModel):
    id: str
    subject: str
    category: str
    priority: str
    status: TicketStatus
    description: str
    submitted_by: str
    submitted_at: datetime
    admin_response: str | None = None
    updated_at: datetime
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None


class SubmissionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: SubmissionType
    submitted_by: str = Field(min_length=1, max_length=255)
    title: str | None = Field(default=None, max_length=255)
    data: dict[str, Any]
    asset_payload: dict[str, Any]
    intake_method: str | None = None
    submission_origin: str | None = None


class SubmissionBulkItemCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=255)
    data: dict[str, Any]
    asset_payload: dict[str, Any]
    intake_method: str | None = None
    submission_origin: str | None = None


class SubmissionBulkCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: SubmissionType
    submitted_by: str = Field(min_length=1, max_length=255)
    items: list[SubmissionBulkItemCreate] = Field(min_length=1, max_length=25)


class SubmissionReviewUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: SubmissionStatus
    admin_notes: str | None = None


class SubmissionRecord(BaseModel):
    id: str
    type: SubmissionType
    status: SubmissionStatus
    submitted_by: str
    submitted_at: datetime
    title: str | None = None
    data: dict[str, Any]
    admin_notes: str | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    created_asset_id: int | None = None
    created_asset_type: SubmissionType | None = None
    error_message: str | None = None


class SubmissionBulkItemResult(BaseModel):
    index: int
    created: bool
    submission_id: str | None = None
    error: str | None = None
    submission: SubmissionRecord | None = None


class SubmissionBulkCreateResult(BaseModel):
    total: int
    created: int
    failed: int
    results: list[SubmissionBulkItemResult]
