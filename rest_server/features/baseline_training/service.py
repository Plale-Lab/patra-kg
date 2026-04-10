from __future__ import annotations

from rest_server.features.baseline_training.models import (
    BaselineTrainingRecommendation,
    BaselineTrainingStubResponse,
    BaselineTrainingStubSummary,
    DemoMetric,
    EvalReportSection,
)
from rest_server.features.dataset_assembly.service import build_dataset_composition_preview
from rest_server.features.intent_schema.models import IntentSchemaResult
from rest_server.features.training_readiness.service import evaluate_training_readiness


def _recommendation_for_task(task_type: str) -> BaselineTrainingRecommendation:
    if task_type == "classification":
        return BaselineTrainingRecommendation(
            model_family="Logistic Regression or Gradient Boosted Trees",
            reason="These are practical baseline models for structured classification and work well for early signal validation.",
            training_scope="Single-table tabular classification baseline",
        )
    if task_type == "regression":
        return BaselineTrainingRecommendation(
            model_family="Linear Regression or Gradient Boosted Trees Regressor",
            reason="These are practical baseline models for structured regression and expose feature behavior early.",
            training_scope="Single-table tabular regression baseline",
        )
    if task_type == "forecasting":
        return BaselineTrainingRecommendation(
            model_family="Simple time-split baseline",
            reason="Forecasting needs leakage-aware splits before richer models are useful.",
            training_scope="Time-aware baseline only after explicit split definition",
        )
    return BaselineTrainingRecommendation(
        model_family="No baseline selected",
        reason="The task type is still ambiguous, so a model family should not be chosen automatically.",
        training_scope="Clarify task type before baseline training",
    )


def _simulated_metrics(task_type: str, *, ready_for_demo: bool) -> list[DemoMetric]:
    if not ready_for_demo:
        return [
            DemoMetric(
                name="Training status",
                value="Not executed",
                description="The stub refused to fabricate baseline metrics because the current plan is blocked or incomplete.",
            )
        ]

    if task_type == "classification":
        return [
            DemoMetric(
                name="Validation AUC",
                value="0.58-0.70 (simulated range)",
                description="Illustrative baseline range for a first-pass structured classification model. This is not measured from a real training run.",
            ),
            DemoMetric(
                name="Validation accuracy",
                value="0.62-0.76 (simulated range)",
                description="Illustrative range only. Final accuracy depends on label balance, leakage checks, and actual dataset assembly quality.",
            ),
        ]
    if task_type == "regression":
        return [
            DemoMetric(
                name="R-squared",
                value="0.20-0.45 (simulated range)",
                description="Illustrative baseline range for a first-pass structured regression model. This is not measured from a real training run.",
            ),
            DemoMetric(
                name="MAE stability",
                value="Needs dataset-specific scaling",
                description="Absolute regression error depends on the target scale and is intentionally not guessed here.",
            ),
        ]
    return [
        DemoMetric(
            name="Training status",
            value="Stub only",
            description="This task type does not yet have a deterministic baseline metric template.",
        )
    ]


def run_baseline_training_stub(
    *,
    intent_schema: IntentSchemaResult,
    top_k: int,
    disable_llm: bool,
    api_base: str | None,
    model: str | None,
    api_key: str | None,
    timeout_seconds: int,
    cache_dir: str | None,
    preview_row_limit: int,
) -> BaselineTrainingStubResponse:
    readiness = evaluate_training_readiness(
        intent_schema=intent_schema,
        top_k=top_k,
        disable_llm=disable_llm,
        api_base=api_base,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        cache_dir=cache_dir,
    )
    preview = build_dataset_composition_preview(
        intent_schema=intent_schema,
        top_k=top_k,
        disable_llm=disable_llm,
        api_base=api_base,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        cache_dir=cache_dir,
        preview_row_limit=preview_row_limit,
    )

    recommendation = _recommendation_for_task(intent_schema.task_type)
    ready_for_demo = readiness.summary.gate_status != "blocked" and not preview.manifest.blocked
    execution_status = "simulated_report" if ready_for_demo else "blocked_stub"

    report_sections: list[EvalReportSection] = [
        EvalReportSection(
            title="Execution mode",
            bullets=[
                "This module is a demo training stub. It does not run a real model fit yet.",
                "All metrics shown below are explicitly marked simulated unless replaced by a later real training module.",
            ],
        ),
        EvalReportSection(
            title="Data composition",
            bullets=[
                f"Selected datasets: {readiness.summary.selected_dataset_count}",
                f"Preview rows available: {len(preview.preview_rows)}",
                f"Direct fields: {readiness.summary.direct_count}",
                f"Derivable fields: {readiness.summary.derivable_count}",
                f"Missing fields: {readiness.summary.missing_count}",
            ],
        ),
    ]

    if readiness.issues:
        report_sections.append(
            EvalReportSection(
                title="Blocking and warning conditions",
                bullets=[f"{item.severity.upper()}: {item.message}" for item in readiness.issues],
            )
        )
    else:
        report_sections.append(
            EvalReportSection(
                title="Readiness status",
                bullets=["The current deterministic plan is sufficiently complete for a demo baseline handoff."],
            )
        )

    report_sections.append(
        EvalReportSection(
            title="Recommended baseline",
            bullets=[
                f"Model family: {recommendation.model_family}",
                f"Scope: {recommendation.training_scope}",
                recommendation.reason,
            ],
        )
    )

    report_sections.append(
        EvalReportSection(
            title="Limits",
            bullets=[
                "No real train/validation/test split was executed.",
                "No real feature engineering or hyperparameter search was executed.",
                "Any downstream demo must keep simulated metrics labeled as simulated.",
            ],
        )
    )

    summary = BaselineTrainingStubSummary(
        execution_status=execution_status,
        gate_status=readiness.summary.gate_status,
        task_type=intent_schema.task_type,
        selected_dataset_count=readiness.summary.selected_dataset_count,
        preview_row_count=len(preview.preview_rows),
        direct_count=readiness.summary.direct_count,
        derivable_count=readiness.summary.derivable_count,
        missing_count=readiness.summary.missing_count,
    )

    message = (
        "Baseline training stub completed with a simulated eval report."
        if ready_for_demo
        else "Baseline training stub stopped at the readiness boundary and returned a blocked demo report."
    )
    return BaselineTrainingStubResponse(
        status="ok",
        message=message,
        summary=summary,
        recommendation=recommendation,
        metrics=_simulated_metrics(intent_schema.task_type, ready_for_demo=ready_for_demo),
        eval_report=report_sections,
        training_readiness=readiness.model_dump(),
        composition_preview=preview.model_dump(),
    )
