from __future__ import annotations

from rest_server.features.dataset_assembly.service import build_dataset_assembly_plan
from rest_server.features.intent_schema.models import IntentSchemaResult
from rest_server.features.training_readiness.models import (
    TrainingGateIssue,
    TrainingReadinessResponse,
    TrainingReadinessSummary,
)


def evaluate_training_readiness(
    *,
    intent_schema: IntentSchemaResult,
    top_k: int,
    disable_llm: bool,
    api_base: str | None,
    model: str | None,
    api_key: str | None,
    timeout_seconds: int,
    cache_dir: str | None,
) -> TrainingReadinessResponse:
    plan = build_dataset_assembly_plan(
        intent_schema=intent_schema,
        top_k=top_k,
        disable_llm=disable_llm,
        api_base=api_base,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        cache_dir=cache_dir,
    )

    issues: list[TrainingGateIssue] = []
    recommendations: list[str] = []

    if plan.summary.missing_count > 0:
        issues.append(
            TrainingGateIssue(
                severity="error",
                code="missing_fields",
                message="One or more target fields have no safe source resolution. Training should remain blocked.",
            )
        )
        recommendations.append("Increase internal column-aware asset coverage before attempting dataset assembly for training.")

    if plan.summary.selected_dataset_count == 0:
        issues.append(
            TrainingGateIssue(
                severity="error",
                code="no_selected_datasets",
                message="No dataset was selected for the current plan.",
            )
        )
        recommendations.append("Improve metadata indexing and internal schema density before training.")

    if plan.summary.requires_manual_join_confirmation:
        issues.append(
            TrainingGateIssue(
                severity="warning",
                code="manual_join_confirmation",
                message="Multiple datasets are required. Join keys and temporal alignment need manual confirmation.",
            )
        )
        recommendations.append("Add explicit join-key review before any assembled dataset is treated as trainable.")

    metadata_only_count = sum(1 for item in plan.selected_datasets if item.source_quality == "metadata_only")
    real_column_aware_dataset_count = sum(
        1 for item in plan.selected_datasets if item.source_tier == "tier1_real" and item.source_quality == "column_aware"
    )
    if metadata_only_count > 0:
        issues.append(
            TrainingGateIssue(
                severity="warning",
                code="metadata_only_sources",
                message="Some selected sources are metadata-only. They are suitable for discovery but not yet strong evidence for training assembly.",
            )
        )
        recommendations.append("Upgrade more internal assets from metadata-only to column-aware before training.")

    if plan.summary.derivable_count > 0:
        issues.append(
            TrainingGateIssue(
                severity="warning",
                code="derivable_fields_present",
                message="Some fields are only marked derivable. Explicit transform logic is still required.",
            )
        )
        recommendations.append("Add deterministic transform templates before treating derivable fields as training-ready.")

    gate_status = "blocked"
    if issues and not any(item.severity == "error" for item in issues):
        gate_status = "warning"
    if not issues:
        gate_status = "ready"

    summary = TrainingReadinessSummary(
        gate_status=gate_status,
        total_fields=plan.summary.total_fields,
        direct_count=plan.summary.direct_count,
        derivable_count=plan.summary.derivable_count,
        missing_count=plan.summary.missing_count,
        selected_dataset_count=plan.summary.selected_dataset_count,
        real_column_aware_dataset_count=real_column_aware_dataset_count,
    )

    if gate_status == "ready":
        recommendations.append("Proceed to deterministic dataset assembly on the selected internal real assets.")

    return TrainingReadinessResponse(
        status="ok",
        message="Training readiness evaluated from the current deterministic assembly plan.",
        summary=summary,
        issues=issues,
        recommendations=recommendations,
        assembly_plan=plan.model_dump(),
    )
