from __future__ import annotations

import os
import re

from rest_server.deps import PatraActor
from rest_server.features.ask_patra.models import (
    AskPatraHandoff,
    AskPatraIntent,
    AskPatraSuggestedAction,
    AskPatraToolCapability,
    AskPatraToolCard,
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() == "true"


TOOL_DEFINITIONS: dict[str, dict] = {
    "browse_model_cards": {
        "title": "Browse Model Cards",
        "domain": "catalog",
        "summary": "Search and open published model cards from the PATRA catalog.",
        "route": "/explore-model-cards",
        "read_only": True,
        "requires_login": False,
        "requires_admin": False,
        "supports_inline": False,
        "enabled": lambda: True,
    },
    "browse_datasheets": {
        "title": "Browse Datasheets",
        "domain": "catalog",
        "summary": "Search and open datasheets from the PATRA catalog.",
        "route": "/explore-datasheets",
        "read_only": True,
        "requires_login": False,
        "requires_admin": False,
        "supports_inline": False,
        "enabled": lambda: True,
    },
    "intent_schema": {
        "title": "Intent Schema",
        "domain": "planning",
        "summary": "Turn a modeling goal into a structured schema draft and planning flow.",
        "route": "/intent-schema",
        "read_only": True,
        "requires_login": True,
        "requires_admin": False,
        "supports_inline": True,
        "enabled": lambda: _env_flag("ENABLE_INTENT_SCHEMA", default=False),
    },
    "mvp_demo_report": {
        "title": "MVP Demo Report",
        "domain": "planning",
        "summary": "Run the current deterministic demo pipeline and summarize the result.",
        "route": "/mvp-demo-report",
        "read_only": True,
        "requires_login": True,
        "requires_admin": False,
        "supports_inline": True,
        "enabled": lambda: _env_flag("ENABLE_MVP_DEMO_REPORT", default=False),
    },
    "agent_tools": {
        "title": "Agent Toolkit",
        "domain": "workflow",
        "summary": "Inspect and run user-facing PATRA tools from the toolkit surface.",
        "route": "/agent-tools",
        "read_only": False,
        "requires_login": True,
        "requires_admin": False,
        "supports_inline": False,
        "enabled": lambda: _env_flag("ENABLE_AGENT_TOOLS", default=True),
    },
    "automated_ingestion": {
        "title": "Automated Ingestion",
        "domain": "workflow",
        "summary": "Stage external CSV sources and review generated ingestion artifacts.",
        "route": "/record-scrape",
        "read_only": False,
        "requires_login": True,
        "requires_admin": True,
        "supports_inline": False,
        "enabled": lambda: _env_flag("ENABLE_AUTOMATED_INGESTION", default=False),
    },
    "edit_records": {
        "title": "Edit Records",
        "domain": "workflow",
        "summary": "Search for an existing model card or datasheet and update it in place.",
        "route": "/edit-records",
        "read_only": False,
        "requires_login": True,
        "requires_admin": False,
        "supports_inline": False,
        "enabled": lambda: True,
    },
    "submit_records": {
        "title": "Submit Records",
        "domain": "workflow",
        "summary": "Create a new model card or datasheet submission.",
        "route": "/submit",
        "read_only": False,
        "requires_login": True,
        "requires_admin": False,
        "supports_inline": False,
        "enabled": lambda: True,
    },
    "tickets": {
        "title": "Tickets",
        "domain": "support",
        "summary": "Open or review support tickets for PATRA workflows.",
        "route": "/tickets",
        "read_only": False,
        "requires_login": True,
        "requires_admin": False,
        "supports_inline": False,
        "enabled": lambda: True,
    },
    "mcp_explorer": {
        "title": "MCP Explorer",
        "domain": "integration",
        "summary": "Inspect connected MCP tools and resources.",
        "route": "/mcp-explorer",
        "read_only": True,
        "requires_login": False,
        "requires_admin": False,
        "supports_inline": True,
        "enabled": lambda: _env_flag("ENABLE_MCP_EXPLORER", default=False),
    },
    "animal_ecology": {
        "title": "Animal Ecology",
        "domain": "experiments",
        "summary": "Review camera-trap and deployment activity for the animal ecology domain.",
        "route": "/animal-ecology",
        "read_only": True,
        "requires_login": False,
        "requires_admin": False,
        "supports_inline": True,
        "enabled": lambda: _env_flag("ENABLE_DOMAIN_EXPERIMENTS", default=False),
    },
    "digital_agriculture": {
        "title": "Digital Agriculture",
        "domain": "experiments",
        "summary": "Review crop-yield and field experiment activity for the digital agriculture domain.",
        "route": "/digital-agriculture",
        "read_only": True,
        "requires_login": False,
        "requires_admin": False,
        "supports_inline": True,
        "enabled": lambda: _env_flag("ENABLE_DOMAIN_EXPERIMENTS", default=False),
    },
}


def _availability_for_actor(definition: dict, actor: PatraActor | None) -> tuple[str, str | None]:
    if not definition["enabled"]():
        return "disabled", "This tool is disabled in the current backend deployment."
    if definition["requires_admin"] and not (actor and actor.is_admin):
        return "admin_only", "This tool requires an admin session."
    if definition["requires_login"] and not (actor and actor.is_authenticated):
        return "requires_login", "This tool requires a signed-in PATRA session."
    return "available", None


def get_tool_capabilities(actor: PatraActor | None = None) -> list[AskPatraToolCapability]:
    capabilities: list[AskPatraToolCapability] = []
    for tool_id, definition in TOOL_DEFINITIONS.items():
        availability, reason = _availability_for_actor(definition, actor)
        capabilities.append(
            AskPatraToolCapability(
                tool_id=tool_id,
                title=definition["title"],
                domain=definition["domain"],
                summary=definition["summary"],
                route=definition["route"],
                read_only=definition["read_only"],
                requires_login=definition["requires_login"],
                requires_admin=definition["requires_admin"],
                supports_inline=definition["supports_inline"],
                availability=availability,
                availability_reason=reason,
            )
        )
    return capabilities


def get_tool_capability_map(actor: PatraActor | None = None) -> dict[str, AskPatraToolCapability]:
    return {cap.tool_id: cap for cap in get_tool_capabilities(actor)}


def classify_tool_intent(message: str) -> AskPatraIntent:
    normalized = re.sub(r"\s+", " ", message.strip().lower())
    if not normalized:
        return AskPatraIntent(category="unknown", confidence=0.0, tool_target=None)

    if any(phrase in normalized for phrase in ("mvp demo", "demo report", "executive summary", "pipeline demo")):
        return AskPatraIntent(category="mvp_demo_report", confidence=0.98, tool_target="mvp_demo_report")
    if any(phrase in normalized for phrase in ("intent schema", "training schema", "target schema", "schema draft", "modeling intent")):
        return AskPatraIntent(category="intent_schema", confidence=0.98, tool_target="intent_schema")
    if "mcp" in normalized or "model context protocol" in normalized:
        return AskPatraIntent(category="mcp_explorer", confidence=0.98, tool_target="mcp_explorer")
    if any(phrase in normalized for phrase in ("animal ecology", "camera trap", "wildlife", "ecology experiment")):
        return AskPatraIntent(category="animal_ecology", confidence=0.98, tool_target="animal_ecology")
    if any(phrase in normalized for phrase in ("digital agriculture", "crop yield", "sugarcane", "digital ag", "field experiment")):
        return AskPatraIntent(category="digital_agriculture", confidence=0.98, tool_target="digital_agriculture")
    if any(phrase in normalized for phrase in ("agent toolkit", "agent tool", "toolkit", "run a tool")):
        return AskPatraIntent(category="agent_tools", confidence=0.95, tool_target="agent_tools")
    if any(phrase in normalized for phrase in ("automated ingestion", "record scrape", "scrape this csv", "ingest this csv", "scrape this source")):
        return AskPatraIntent(category="automated_ingestion", confidence=0.97, tool_target="automated_ingestion")
    if any(phrase in normalized for phrase in ("edit this model card", "edit this datasheet", "edit record", "update this model card", "update this datasheet")):
        return AskPatraIntent(category="edit_records", confidence=0.97, tool_target="edit_records")
    if any(phrase in normalized for phrase in ("submit a dataset", "submit a model card", "submit this dataset", "submit record", "new submission")):
        return AskPatraIntent(category="submit_records", confidence=0.97, tool_target="submit_records")
    if any(phrase in normalized for phrase in ("open a ticket", "create a ticket", "support ticket", "bug report", "feature request", "access request")):
        return AskPatraIntent(category="tickets", confidence=0.97, tool_target="tickets")
    if any(phrase in normalized for phrase in ("browse datasheets", "open datasheets", "show datasheets")):
        return AskPatraIntent(category="browse_datasheets", confidence=0.94, tool_target="browse_datasheets")
    if any(phrase in normalized for phrase in ("browse model cards", "open model cards", "show model cards")):
        return AskPatraIntent(category="browse_model_cards", confidence=0.94, tool_target="browse_model_cards")
    if "experiment" in normalized:
        return AskPatraIntent(category="experiments", confidence=0.8, tool_target=None)
    if any(phrase in normalized for phrase in ("what can you help", "what can patra do", "what can you do", "help me navigate")):
        return AskPatraIntent(category="capability", confidence=0.9, tool_target=None)
    return AskPatraIntent(category="unknown", confidence=0.0, tool_target=None)


def build_tool_navigation(
    *,
    tool_id: str,
    actor: PatraActor | None,
    reason: str,
    label: str | None = None,
    query: dict[str, str] | None = None,
    prefilled_payload: dict | None = None,
    cta_kind: str = "navigate",
) -> tuple[AskPatraToolCard, AskPatraSuggestedAction, AskPatraHandoff]:
    capability = get_tool_capability_map(actor).get(tool_id)
    if capability is None:
        raise KeyError(f"Unknown tool id: {tool_id}")
    action = AskPatraSuggestedAction(
        action_id=f"{tool_id}:{cta_kind}",
        label=label or f"Open {capability.title}",
        route=capability.route,
        query=query or {},
        prefilled_payload=prefilled_payload or {},
        availability=capability.availability,
        reason=capability.availability_reason or reason,
        cta_kind="disabled" if capability.availability != "available" else cta_kind,
    )
    card = AskPatraToolCard(
        tool_id=capability.tool_id,
        title=capability.title,
        domain=capability.domain,
        summary=capability.summary,
        reason=reason,
        availability=capability.availability,
        read_only=capability.read_only,
        supports_inline=capability.supports_inline,
        route=capability.route,
        cta_label=action.label,
    )
    handoff = AskPatraHandoff(
        kind="navigate" if cta_kind == "navigate" else "prefill",
        tool_target=capability.tool_id,
        route=capability.route,
        label=action.label,
        prefilled_payload=action.prefilled_payload,
    )
    return card, action, handoff
