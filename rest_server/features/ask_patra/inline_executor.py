from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import dataclass
from typing import Awaitable, Callable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import asyncpg

from rest_server.features.ask_patra.models import AskPatraExecution, AskPatraIntent
from rest_server.features.intent_schema.service import generate_schema
from rest_server.features.mvp_demo_report.service import build_mvp_demo_report
from rest_server.routes.experiments import DOMAIN_TABLES


@dataclass(slots=True)
class InlineExecutionContext:
    tool_id: str
    tool_route: str | None
    message: str | None
    context: str | None
    query: dict[str, str]
    prefilled_payload: dict
    disable_llm: bool
    request_tapis_token: str | None
    pool: asyncpg.Pool | None


@dataclass(slots=True)
class InlineExecutionOutcome:
    content: str
    execution: AskPatraExecution
    intent: AskPatraIntent
    navigation_reason: str
    navigation_query: dict[str, str]
    handoff_label: str


InlineExecutor = Callable[[InlineExecutionContext], Awaitable[InlineExecutionOutcome]]


def _coerce_int(value, default: int, *, minimum: int, maximum: int) -> int:
    with suppress(TypeError, ValueError):
        coerced = int(value)
        return max(minimum, min(maximum, coerced))
    return default


def _mcp_base_url() -> str:
    return os.getenv("MCP_BASE_URL", "http://127.0.0.1:8050").strip() or "http://127.0.0.1:8050"


def _read_mcp_endpoint(base_url: str) -> tuple[str | None, str | None]:
    request = Request(f"{base_url.rstrip('/')}/sse", headers={"Accept": "text/event-stream"})
    try:
        with urlopen(request, timeout=5) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if line.startswith("data:"):
                    return line[5:].strip(), None
    except Exception as exc:
        return None, str(exc)
    return None, "MCP endpoint did not emit an SSE data line."


def _post_mcp_rpc(endpoint_url: str, payload: dict) -> dict:
    request = Request(
        endpoint_url,
        data=__import__("json").dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        raw_body = response.read().decode("utf-8")
    if not raw_body:
        return {}
    return __import__("json").loads(raw_body)


def _run_mcp_preview() -> dict:
    base_url = _mcp_base_url()
    endpoint_data, endpoint_error = _read_mcp_endpoint(base_url)
    if endpoint_error:
        return {
            "kind": "mcp_explorer",
            "connected": False,
            "endpoint_url": None,
            "tool_count": 0,
            "tools": [],
            "error": endpoint_error,
            "mcp_base_url": base_url,
        }
    endpoint_url = urljoin(f"{base_url.rstrip('/')}/", endpoint_data or "")
    initialize_result = _post_mcp_rpc(
        endpoint_url,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ask-patra-inline-preview", "version": "1.0"},
            },
        },
    )
    _post_mcp_rpc(endpoint_url, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
    tools_result = _post_mcp_rpc(endpoint_url, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    tools = (tools_result or {}).get("result", {}).get("tools", []) or []
    server_name = (initialize_result or {}).get("result", {}).get("serverInfo", {}).get("name")
    return {
        "kind": "mcp_explorer",
        "connected": True,
        "endpoint_url": endpoint_url,
        "mcp_base_url": base_url,
        "server_name": server_name,
        "tool_count": len(tools),
        "tools": [tool.get("name") for tool in tools if tool.get("name")][:8],
    }


def _build_mcp_execution_payload(result: dict) -> dict:
    return {
        "kind": "mcp_explorer",
        "connected": bool(result.get("connected")),
        "server_name": result.get("server_name"),
        "tool_count": int(result.get("tool_count") or 0),
        "tools": result.get("tools") or [],
        "endpoint_url": result.get("endpoint_url"),
        "mcp_base_url": result.get("mcp_base_url"),
        "error": result.get("error"),
    }


def _format_mcp_execution_message(result: dict) -> str:
    if not result.get("connected"):
        return (
            "**MCP preview failed.**\n"
            f"- Endpoint: `{result.get('mcp_base_url') or 'not configured'}`\n"
            f"- Error: {result.get('error') or 'Unknown connection failure.'}\n\n"
            "Open MCP Explorer to inspect the full integration surface once the endpoint is available."
        )
    tools = result.get("tools") or []
    preview = "\n".join(f"- `{tool}`" for tool in tools[:5]) or "- No tools reported"
    return (
        "**MCP preview generated.**\n"
        f"- Server: **{result.get('server_name') or 'unknown'}**\n"
        f"- Tool count: **{result.get('tool_count') or 0}**\n"
        f"{preview}\n\n"
        "Open MCP Explorer for the full interactive tool surface."
    )


def _build_intent_schema_execution_payload(result) -> dict:
    return {
        "kind": "intent_schema",
        "mode": result.mode,
        "provider": result.provider,
        "model_used": result.model_used,
        "task_type": result.task_type,
        "entity_grain": result.entity_grain,
        "target_column": result.target_column,
        "field_count": len(result.schema_fields),
        "assumption_count": len(result.assumptions),
        "ambiguity_warning_count": len(result.ambiguity_warnings),
        "fields": [
            {
                "name": field.name,
                "type": field.data_type,
                "role": field.semantic_role,
                "required": field.required,
            }
            for field in result.schema_fields
        ],
    }


def _format_intent_schema_execution_message(result) -> str:
    field_count = len(result.schema_fields)
    warning_count = len(result.ambiguity_warnings)
    return (
        "**Intent Schema generated inline.**\n"
        f"- Task type: **{result.task_type}**\n"
        f"- Target column: **{result.target_column}**\n"
        f"- Fields: **{field_count}**\n"
        f"- Ambiguity warnings: **{warning_count}**\n\n"
        "Open the full Intent Schema surface if you want to inspect every field, assumption, and warning."
    )


def _build_mvp_demo_execution_payload(report) -> dict:
    training_summary = (report.training_stub or {}).get("summary") or {}
    preview_rows = (report.composition_preview or {}).get("preview_rows") or []
    return {
        "kind": "mvp_demo_report",
        "schema_mode": report.schema.mode,
        "discovery_pool_mode": report.discovery.pool_mode,
        "candidate_dataset_count": report.discovery.candidate_dataset_count,
        "selected_dataset_count": report.assembly_plan.summary.selected_dataset_count,
        "training_gate_status": training_summary.get("gate_status"),
        "execution_status": training_summary.get("execution_status"),
        "preview_row_count": len(preview_rows),
        "preview_rows": preview_rows[:5],
    }


def _format_mvp_demo_execution_message(report) -> str:
    training_summary = (report.training_stub or {}).get("summary") or {}
    gate_status = str(training_summary.get("gate_status") or "unknown")
    execution_status = str(training_summary.get("execution_status") or "unknown")
    preview_row_count = len((report.composition_preview or {}).get("preview_rows") or [])
    return (
        "**MVP Demo Report generated in deterministic mode.**\n"
        f"- Training gate: **{gate_status}**\n"
        f"- Stub execution: **{execution_status}**\n"
        f"- Preview rows: **{preview_row_count}**\n\n"
        "Open the full MVP Demo Report surface if you want the full executive summary and raw JSON."
    )


async def _build_experiment_preview(*, pool: asyncpg.Pool, tool_id: str) -> dict:
    domain = "animal-ecology" if tool_id == "animal_ecology" else "digital-ag"
    tables = DOMAIN_TABLES.get(domain)
    if not tables:
        return {"kind": tool_id, "domain": domain, "available": False, "error": "Unknown experiment domain."}
    events_table = tables["events"]
    async with pool.acquire() as conn:
        user_rows = await conn.fetch(f"SELECT DISTINCT user_id FROM {events_table} ORDER BY user_id LIMIT 5")
        summary_rows = await conn.fetch(
            f"""
            SELECT
                experiment_id,
                user_id,
                model_id,
                MAX(total_images) AS total_images,
                MAX(precision) AS precision,
                MAX(recall) AS recall,
                MAX(f1_score) AS f1_score
            FROM {events_table}
            GROUP BY experiment_id, user_id, model_id
            ORDER BY experiment_id
            LIMIT 5
            """
        )
        total_rows = await conn.fetchval(f"SELECT COUNT(*) FROM {events_table}")
    summaries = [
        {
            "experiment_id": str(row["experiment_id"]),
            "user_id": str(row["user_id"]),
            "model_id": str(row["model_id"]),
            "total_images": int(row["total_images"] or 0),
            "precision": float(row["precision"] or 0),
            "recall": float(row["recall"] or 0),
            "f1_score": float(row["f1_score"] or 0),
        }
        for row in summary_rows
    ]
    return {
        "kind": tool_id,
        "domain": domain,
        "available": True,
        "user_count": len(user_rows),
        "total_rows": int(total_rows or 0),
        "users": [str(row["user_id"]) for row in user_rows],
        "experiments": summaries,
    }


def _format_experiment_execution_message(result: dict) -> str:
    if not result.get("available"):
        return f"**Experiment preview failed.** {result.get('error') or 'Unknown error.'}"
    label = "Animal Ecology" if result.get("domain") == "animal-ecology" else "Digital Agriculture"
    top = (result.get("experiments") or [])[:2]
    top_lines = [f"- **{item['experiment_id']}** via `{item['model_id']}` ({item.get('total_images') or 0} images)" for item in top]
    details = "\n".join(top_lines) if top_lines else "- No experiment summaries available"
    return (
        f"**{label} preview generated.**\n"
        f"- Users indexed: **{result.get('user_count') or 0}**\n"
        f"- Event rows: **{result.get('total_rows') or 0}**\n"
        f"{details}\n\n"
        "Open the full experiments page for user selection, detailed metrics, and image traces."
    )


async def _run_intent_schema_tool(context: InlineExecutionContext) -> InlineExecutionOutcome:
    if not context.message:
        raise ValueError("Intent Schema needs a modeling intent before it can run inline.")
    max_fields = _coerce_int(context.prefilled_payload.get("max_fields") or context.query.get("max_fields"), 8, minimum=3, maximum=20)
    result = generate_schema(
        intent_text=context.message,
        context=context.context,
        max_fields=max_fields,
        disable_llm=context.disable_llm,
        request_tapis_token=context.request_tapis_token,
    )
    return InlineExecutionOutcome(
        content=_format_intent_schema_execution_message(result),
        execution=AskPatraExecution(
            state="succeeded",
            message="Intent Schema finished in deterministic inline mode.",
            next_step_route=context.tool_route,
            tool_id=context.tool_id,
            result=_build_intent_schema_execution_payload(result),
        ),
        intent=AskPatraIntent(category="intent_schema", confidence=1.0, tool_target=context.tool_id),
        navigation_reason="Open the full planning surface to inspect and refine the schema.",
        navigation_query={"intent": context.message},
        handoff_label="Intent Schema ran inline",
    )


async def _run_mvp_demo_tool(context: InlineExecutionContext) -> InlineExecutionOutcome:
    if not context.message:
        raise ValueError("MVP Demo Report needs a modeling intent before it can run inline.")
    max_fields = _coerce_int(context.prefilled_payload.get("max_fields") or context.query.get("max_fields"), 8, minimum=3, maximum=20)
    top_k = _coerce_int(context.prefilled_payload.get("top_k") or context.query.get("top_k"), 5, minimum=1, maximum=20)
    preview_row_limit = _coerce_int(
        context.prefilled_payload.get("preview_row_limit") or context.query.get("preview_row_limit"),
        5,
        minimum=1,
        maximum=20,
    )
    report = build_mvp_demo_report(
        intent_text=context.message,
        context=context.context,
        max_fields=max_fields,
        top_k=top_k,
        disable_llm=context.disable_llm,
        preview_row_limit=preview_row_limit,
    )
    return InlineExecutionOutcome(
        content=_format_mvp_demo_execution_message(report),
        execution=AskPatraExecution(
            state="succeeded",
            message="MVP Demo Report finished in deterministic inline mode.",
            next_step_route=context.tool_route,
            tool_id=context.tool_id,
            result=_build_mvp_demo_execution_payload(report),
        ),
        intent=AskPatraIntent(category="mvp_demo_report", confidence=1.0, tool_target=context.tool_id),
        navigation_reason="Open the full demo surface to inspect the full executive summary.",
        navigation_query={"intent": context.message},
        handoff_label="MVP Demo ran inline",
    )


async def _run_mcp_explorer_tool(context: InlineExecutionContext) -> InlineExecutionOutcome:
    try:
        result = _run_mcp_preview()
    except Exception as exc:
        result = {
            "kind": "mcp_explorer",
            "connected": False,
            "endpoint_url": None,
            "tool_count": 0,
            "tools": [],
            "error": str(exc),
            "mcp_base_url": _mcp_base_url(),
        }
    return InlineExecutionOutcome(
        content=_format_mcp_execution_message(result),
        execution=AskPatraExecution(
            state="succeeded" if result.get("connected") else "failed",
            message="MCP preview completed." if result.get("connected") else "MCP preview could not connect to the configured endpoint.",
            next_step_route=context.tool_route,
            tool_id=context.tool_id,
            result=_build_mcp_execution_payload(result),
        ),
        intent=AskPatraIntent(category="mcp_explorer", confidence=1.0, tool_target=context.tool_id),
        navigation_reason="Open MCP Explorer for the full interactive tool surface.",
        navigation_query={},
        handoff_label="MCP preview ran inline",
    )


async def _run_experiment_tool(context: InlineExecutionContext) -> InlineExecutionOutcome:
    if context.pool is None:
        raise ValueError("Experiment preview requires a backend database connection pool.")
    result = await _build_experiment_preview(pool=context.pool, tool_id=context.tool_id)
    return InlineExecutionOutcome(
        content=_format_experiment_execution_message(result),
        execution=AskPatraExecution(
            state="succeeded" if result.get("available") else "failed",
            message="Experiment preview completed." if result.get("available") else "Experiment preview could not be generated.",
            next_step_route=context.tool_route,
            tool_id=context.tool_id,
            result=result,
        ),
        intent=AskPatraIntent(category=context.tool_id, confidence=1.0, tool_target=context.tool_id),
        navigation_reason="Open the full experiments surface for detailed telemetry and images.",
        navigation_query={},
        handoff_label="Experiment preview ran inline",
    )


INLINE_EXECUTOR_REGISTRY: dict[str, InlineExecutor] = {
    "intent_schema": _run_intent_schema_tool,
    "mvp_demo_report": _run_mvp_demo_tool,
    "mcp_explorer": _run_mcp_explorer_tool,
    "animal_ecology": _run_experiment_tool,
    "digital_agriculture": _run_experiment_tool,
}


def get_inline_executor_registry() -> dict[str, InlineExecutor]:
    return INLINE_EXECUTOR_REGISTRY


async def run_inline_executor(context: InlineExecutionContext) -> InlineExecutionOutcome:
    executor = INLINE_EXECUTOR_REGISTRY.get(context.tool_id)
    if executor is None:
        raise ValueError("No inline executor is implemented for this tool.")
    return await executor(context)
