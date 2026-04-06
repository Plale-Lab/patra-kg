from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AskPatraStarter(BaseModel):
    title: str
    prompt: str


class AskPatraCitation(BaseModel):
    resource_type: Literal["model_card", "datasheet"]
    resource_id: int
    title: str
    subtitle: str | None = None
    description: str | None = None
    route: str
    matched_on: list[str] = Field(default_factory=list)


class AskPatraMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str


class AskPatraChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = Field(default=None, max_length=128)
    reset: bool = False


class AskPatraChatResponse(BaseModel):
    conversation_id: str
    answer: str
    mode: Literal["llm", "code_fallback"]
    provider: str
    model_used: str | None = None
    citations: list[AskPatraCitation] = Field(default_factory=list)
    messages: list[AskPatraMessage] = Field(default_factory=list)
    starter_prompts: list[AskPatraStarter] = Field(default_factory=list)


class AskPatraBootstrapResponse(BaseModel):
    enabled: bool
    provider: str
    starter_prompts: list[AskPatraStarter] = Field(default_factory=list)

