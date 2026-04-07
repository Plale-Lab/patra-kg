from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
