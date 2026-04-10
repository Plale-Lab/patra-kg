from typing import Any, Literal

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


class AskPatraIntent(BaseModel):
    category: Literal[
        "greeting",
        "capability",
        "record_search",
        "browse_model_cards",
        "browse_datasheets",
        "intent_schema",
        "mvp_demo_report",
        "agent_tools",
        "automated_ingestion",
        "edit_records",
        "submit_records",
        "tickets",
        "mcp_explorer",
        "animal_ecology",
        "digital_agriculture",
        "experiments",
        "general_help",
        "unknown",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    tool_target: str | None = None


class AskPatraSuggestedAction(BaseModel):
    action_id: str
    label: str
    route: str | None = None
    query: dict[str, str] = Field(default_factory=dict)
    prefilled_payload: dict[str, Any] = Field(default_factory=dict)
    availability: Literal["available", "requires_login", "admin_only", "disabled"] = "available"
    reason: str | None = None
    cta_kind: Literal["navigate", "prefill", "inline", "disabled"] = "navigate"


class AskPatraToolCard(BaseModel):
    tool_id: str
    title: str
    domain: str
    summary: str
    reason: str
    availability: Literal["available", "requires_login", "admin_only", "disabled"] = "available"
    read_only: bool = True
    supports_inline: bool = False
    route: str | None = None
    cta_label: str | None = None


class AskPatraHandoff(BaseModel):
    kind: Literal["explanatory", "navigate", "prefill", "inline"] = "explanatory"
    tool_target: str | None = None
    route: str | None = None
    label: str | None = None
    prefilled_payload: dict[str, Any] = Field(default_factory=dict)


class AskPatraExecution(BaseModel):
    state: Literal["idle", "running", "blocked", "succeeded", "failed"] = "idle"
    message: str | None = None
    next_step_route: str | None = None
    tool_id: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class AskPatraToolCapability(BaseModel):
    tool_id: str
    title: str
    domain: str
    summary: str
    route: str | None = None
    read_only: bool = True
    requires_login: bool = False
    requires_admin: bool = False
    supports_inline: bool = False
    availability: Literal["available", "requires_login", "admin_only", "disabled"] = "available"
    availability_reason: str | None = None


class AskPatraMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str
    intent: AskPatraIntent | None = None
    suggested_actions: list[AskPatraSuggestedAction] = Field(default_factory=list)
    tool_cards: list[AskPatraToolCard] = Field(default_factory=list)
    handoff: AskPatraHandoff | None = None
    execution: AskPatraExecution | None = None


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
    intent: AskPatraIntent | None = None
    suggested_actions: list[AskPatraSuggestedAction] = Field(default_factory=list)
    tool_cards: list[AskPatraToolCard] = Field(default_factory=list)
    handoff: AskPatraHandoff | None = None
    execution: AskPatraExecution | None = None


class AskPatraBootstrapResponse(BaseModel):
    enabled: bool
    provider: str
    starter_prompts: list[AskPatraStarter] = Field(default_factory=list)
    tool_capabilities: list[AskPatraToolCapability] = Field(default_factory=list)


class AskPatraExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str | None = Field(default=None, max_length=128)
    tool_id: str = Field(min_length=1, max_length=64)
    message: str | None = Field(default=None, max_length=8000)
    query: dict[str, str] = Field(default_factory=dict)
    prefilled_payload: dict[str, Any] = Field(default_factory=dict)
    disable_llm: bool = True


class AskPatraExecuteResponse(BaseModel):
    conversation_id: str
    messages: list[AskPatraMessage] = Field(default_factory=list)
    execution: AskPatraExecution

