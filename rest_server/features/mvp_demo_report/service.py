from __future__ import annotations

from rest_server.features.baseline_training.service import run_baseline_training_stub
from rest_server.features.metadata_discovery.service import discover_metadata
from rest_server.features.mvp_demo_report.models import (
    ExecutiveSummaryItem,
    MvpDemoReportResponse,
)
from rest_server.features.intent_schema.service import generate_schema


def build_mvp_demo_report(
    *,
    intent_text: str,
    context: str | None,
    max_fields: int,
    top_k: int,
    disable_llm: bool,
    preview_row_limit: int,
) -> MvpDemoReportResponse:
    schema_result = generate_schema(
        intent_text=intent_text,
        context=context,
        max_fields=max_fields,
        disable_llm=disable_llm,
    )
    metadata_discovery = discover_metadata(
        intent_schema=schema_result,
        top_k=top_k,
        disable_llm=disable_llm,
        api_base=None,
        model=None,
        api_key=None,
        timeout_seconds=60,
        cache_dir=None,
    )
    training_stub = run_baseline_training_stub(
        intent_schema=schema_result,
        top_k=top_k,
        disable_llm=disable_llm,
        api_base=None,
        model=None,
        api_key=None,
        timeout_seconds=60,
        cache_dir=None,
        preview_row_limit=preview_row_limit,
    )

    assembly_plan = training_stub.training_readiness.get("assembly_plan", {})
    composition_preview = training_stub.composition_preview

    discovery_count = len(metadata_discovery.get("ranking") or [])
    top_match = (metadata_discovery.get("top_match") or {}).get("title") or "None"
    gate_status = training_stub.summary.gate_status
    execution_status = training_stub.summary.execution_status
    preview_rows = training_stub.summary.preview_row_count
    direct_count = training_stub.summary.direct_count
    missing_count = training_stub.summary.missing_count

    executive_summary = [
        ExecutiveSummaryItem(label="Task Type", value=schema_result.task_type),
        ExecutiveSummaryItem(label="Target Column", value=schema_result.target_column),
        ExecutiveSummaryItem(label="Candidate Datasets", value=str(discovery_count)),
        ExecutiveSummaryItem(label="Top Match", value=top_match),
        ExecutiveSummaryItem(
            label="Training Gate",
            value=gate_status,
            tone="good" if gate_status == "ready" else ("warn" if gate_status == "warning" else "bad"),
        ),
        ExecutiveSummaryItem(
            label="Stub Execution",
            value=execution_status,
            tone="good" if execution_status == "simulated_report" else "warn",
        ),
        ExecutiveSummaryItem(label="Preview Rows", value=str(preview_rows)),
        ExecutiveSummaryItem(
            label="Field Coverage",
            value=f"{direct_count} direct / {missing_count} missing",
            tone="good" if missing_count == 0 else "warn",
        ),
    ]

    return MvpDemoReportResponse(
        status="ok",
        message="MVP demo report generated from the current deterministic pipeline.",
        executive_summary=executive_summary,
        schema_result=schema_result.model_dump(),
        metadata_discovery=metadata_discovery,
        assembly_plan=assembly_plan,
        composition_preview=composition_preview,
        training_stub=training_stub.model_dump(),
    )
