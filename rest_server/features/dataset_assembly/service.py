from __future__ import annotations

from datetime import datetime
from typing import Any

from rest_server.features.dataset_assembly.models import (
    AssemblyPlanSummary,
    CompositionManifest,
    CompositionManifestField,
    DatasetCompositionPreviewResponse,
    DatasetAssemblyPlanResponse,
    FieldSourceResolution,
    JoinRequirement,
    SelectedDatasetSummary,
)
from rest_server.features.metadata_discovery.service import discover_metadata, intent_schema_to_query_schema
from rest_server.features.metadata_discovery.internal_asset_pool import (
    InternalAssetSchemaPair,
    load_materialized_internal_asset_schema_pool,
)
from rest_server.features.intent_schema.models import IntentSchemaResult
from rest_server.patra_agent_service import _pair_map


def _source_tier_rank(source_tier: str | None, *, source_family: str | None = None) -> int:
    lowered = (source_tier or "").strip().lower()
    if lowered == "tier1_real":
        return 0
    if lowered in {"tier2_approved_derived", "tier2_approved_synthetic", "tier2_approved"}:
        return 1
    if lowered in {"tier3_generated", "tier3_newly_generated"}:
        return 2
    if source_family and source_family.startswith("patra_internal"):
        return 0
    return 3


def _pair_lookup(*, pool_mode: str, cache_dir: str | None) -> dict[str, InternalAssetSchemaPair | Any]:
    if pool_mode == "internal":
        pairs = load_materialized_internal_asset_schema_pool(cache_dir=cache_dir, rebuild_if_missing=True)
        return {pair.dataset_id: pair for pair in pairs}
    return _pair_map(cache_dir)


def _candidate_sort_key(candidate: dict[str, Any], pair_lookup: dict[str, Any]) -> tuple[Any, ...]:
    pair = pair_lookup.get(candidate["dataset_id"])
    source_tier = None
    column_aware = False
    approved = False
    if pair is not None:
        source_tier = getattr(pair, "task_tags", {}).get("source_tier")
        column_aware = bool(getattr(pair, "task_tags", {}).get("column_aware"))
        approved = bool(getattr(pair, "task_tags", {}).get("approved"))
    return (
        _source_tier_rank(source_tier, source_family=candidate.get("source_family")),
        0 if column_aware else 1,
        0 if approved else 1,
        -float(candidate.get("score") or 0.0),
        int(candidate.get("rank") or 9999),
    )


def _direct_source_column(field_name: str, candidate: dict[str, Any], pair: Any) -> str | None:
    for aligned in candidate.get("aligned_pairs") or []:
        target_name = aligned.get("target_field") or aligned.get("query_field") or aligned.get("target_group")
        if target_name == field_name:
            return aligned.get("candidate_field") or aligned.get("source_field") or aligned.get("dataset_field")
    properties = getattr(pair, "schema", {}).get("properties", {})
    if field_name in properties:
        property_payload = properties.get(field_name) or {}
        return property_payload.get("x-patra-source-name") or field_name
    return None


def _derivable_source_columns(field_name: str, candidate: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for support in candidate.get("derived_support") or []:
        target_name = support.get("target_field") or support.get("field") or support.get("target_group")
        if target_name != field_name:
            continue
        source_fields = support.get("source_fields") or support.get("candidate_fields") or []
        for item in source_fields:
            if item and item not in columns:
                columns.append(str(item))
    return columns


def _field_resolution(field_name: str, semantic_role: str, ranked_candidates: list[dict[str, Any]], pair_lookup: dict[str, Any]) -> FieldSourceResolution:
    for candidate in ranked_candidates:
        pair = pair_lookup.get(candidate["dataset_id"])
        task_tags = getattr(pair, "task_tags", {}) if pair is not None else {}
        source_tier = task_tags.get("source_tier") or ("external_fallback" if candidate.get("source_family") != "patra_internal" else "tier1_real")
        source_quality = "column_aware" if task_tags.get("column_aware") else ("metadata_only" if task_tags else None)
        source_approved = bool(task_tags.get("approved")) if task_tags else None

        if field_name in (candidate.get("matched_field_groups") or []):
            source_column = _direct_source_column(field_name, candidate, pair)
            return FieldSourceResolution(
                target_field=field_name,
                semantic_role=semantic_role,
                resolution_status="direct",
                source_tier=source_tier,
                source_quality=source_quality,
                source_approved=source_approved,
                source_dataset_id=candidate["dataset_id"],
                source_dataset_title=candidate.get("title"),
                source_family=candidate.get("source_family"),
                source_url=candidate.get("source_url"),
                source_column=source_column,
                source_columns=[source_column] if source_column else [],
                rationale="Field is directly available from the selected candidate dataset.",
                notes=[f"Candidate score: {candidate.get('score', 0):.3f}"],
            )

        if field_name in (candidate.get("derivable_field_groups") or []):
            source_columns = _derivable_source_columns(field_name, candidate)
            return FieldSourceResolution(
                target_field=field_name,
                semantic_role=semantic_role,
                resolution_status="derivable",
                source_tier=source_tier,
                source_quality=source_quality,
                source_approved=source_approved,
                source_dataset_id=candidate["dataset_id"],
                source_dataset_title=candidate.get("title"),
                source_family=candidate.get("source_family"),
                source_url=candidate.get("source_url"),
                source_columns=source_columns,
                rationale="Field is not directly present but was marked derivable by the shared matcher workflow.",
                notes=[f"Candidate score: {candidate.get('score', 0):.3f}"],
            )

    return FieldSourceResolution(
        target_field=field_name,
        semantic_role=semantic_role,
        resolution_status="missing",
        rationale="No safe direct or derivable source was found in the current ranked candidate pool.",
        notes=["Requires additional internal assets, schema records, or later controlled generation."],
    )


def build_dataset_assembly_plan(
    *,
    intent_schema: IntentSchemaResult,
    top_k: int,
    disable_llm: bool,
    api_base: str | None,
    model: str | None,
    api_key: str | None,
    timeout_seconds: int,
    cache_dir: str | None,
) -> DatasetAssemblyPlanResponse:
    discovery = discover_metadata(
        intent_schema=intent_schema,
        top_k=top_k,
        disable_llm=disable_llm,
        api_base=api_base,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        cache_dir=cache_dir,
    )

    query_schema = intent_schema_to_query_schema(intent_schema)
    pool_mode = discovery.get("pool_mode") or "public"
    pair_lookup = _pair_lookup(pool_mode=pool_mode, cache_dir=cache_dir)
    ranked_candidates = sorted(discovery.get("ranking") or [], key=lambda item: _candidate_sort_key(item, pair_lookup))

    field_resolutions = [
        _field_resolution(field.name, field.semantic_role, ranked_candidates, pair_lookup)
        for field in intent_schema.schema_fields
    ]

    selected_dataset_map: dict[str, dict[str, Any]] = {}
    for resolution in field_resolutions:
        if not resolution.source_dataset_id:
            continue
        selected = selected_dataset_map.setdefault(
            resolution.source_dataset_id,
            {
                "dataset_id": resolution.source_dataset_id,
                "title": resolution.source_dataset_title or resolution.source_dataset_id,
                "source_tier": resolution.source_tier or "unknown",
                "source_quality": resolution.source_quality,
                "approved": bool(resolution.source_approved),
                "source_family": resolution.source_family or "",
                "source_url": resolution.source_url or "",
                "direct_fields": [],
                "derivable_fields": [],
            },
        )
        if resolution.resolution_status == "direct":
            selected["direct_fields"].append(resolution.target_field)
        elif resolution.resolution_status == "derivable":
            selected["derivable_fields"].append(resolution.target_field)

    selected_datasets = [
        SelectedDatasetSummary(**payload)
        for payload in sorted(
            selected_dataset_map.values(),
            key=lambda item: (
                _source_tier_rank(item["source_tier"], source_family=item["source_family"]),
                -(len(item["direct_fields"]) + len(item["derivable_fields"])),
                item["title"].lower(),
            ),
        )
    ]

    direct_count = sum(1 for item in field_resolutions if item.resolution_status == "direct")
    derivable_count = sum(1 for item in field_resolutions if item.resolution_status == "derivable")
    missing_count = sum(1 for item in field_resolutions if item.resolution_status == "missing")

    join_requirements: list[JoinRequirement] = []
    if len(selected_datasets) > 1:
        join_requirements.append(
            JoinRequirement(
                level="warning",
                message="Multiple datasets are needed. Join keys and temporal alignment must be confirmed manually before assembly.",
                datasets=[item.dataset_id for item in selected_datasets],
            )
        )
    elif len(selected_datasets) == 1 and derivable_count > 0:
        join_requirements.append(
            JoinRequirement(
                level="info",
                message="One dataset covers the current plan, but derivable fields still require explicit transformation logic.",
                datasets=[selected_datasets[0].dataset_id],
            )
        )
    if missing_count > 0:
        join_requirements.append(
            JoinRequirement(
                level="warning",
                message="Some target fields remain unresolved. The current plan is incomplete and should not proceed to training without additional sources or later-stage generation policy.",
                datasets=[],
            )
        )

    fallback_recommendations: list[str] = []
    if missing_count > 0:
        fallback_recommendations.append("Add more PATRA internal column-aware assets before relying on external fallback or generation.")
    if pool_mode == "public":
        fallback_recommendations.append("No internal asset pool was available, so this plan was built from the shared public fallback pool.")

    summary = AssemblyPlanSummary(
        direct_count=direct_count,
        derivable_count=derivable_count,
        missing_count=missing_count,
        total_fields=len(field_resolutions),
        selected_dataset_count=len(selected_datasets),
        requires_manual_join_confirmation=len(selected_datasets) > 1,
    )

    return DatasetAssemblyPlanResponse(
        status="ok",
        message="Tiered source resolution completed. Real internal assets remain the highest-priority inputs for assembly planning.",
        query_schema=query_schema,
        pool_mode=pool_mode,
        summary=summary,
        selected_datasets=selected_datasets,
        field_resolutions=field_resolutions,
        join_requirements=join_requirements,
        fallback_recommendations=fallback_recommendations,
    )


def _parse_date_for_preview(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    candidate = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return None


def _preview_value_from_resolution(resolution: FieldSourceResolution, row: dict[str, Any]) -> tuple[bool, Any, str | None]:
    if resolution.resolution_status == "direct":
        source = resolution.source_column or (resolution.source_columns[0] if resolution.source_columns else None)
        if source:
            return True, row.get(source), None
        return False, None, "No direct source column was available for preview."

    if resolution.resolution_status == "derivable":
        sources = resolution.source_columns or []
        if resolution.target_field.lower() == "year" and sources:
            parsed = _parse_date_for_preview(row.get(sources[0]))
            if parsed is not None:
                return True, parsed.year, None
            return False, None, "Could not parse a date value for year extraction."
        if resolution.target_field in {"LAT", "LON", "yield"} and len(sources) == 1:
            return True, row.get(sources[0]), None
        return False, None, "Derivable field requires a deterministic transform that is not yet executed in preview mode."

    return False, None, "Field is unresolved and cannot be included in preview."


def build_dataset_composition_preview(
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
) -> DatasetCompositionPreviewResponse:
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

    selected_dataset_ids = [item.dataset_id for item in plan.selected_datasets]
    manifest_fields: list[CompositionManifestField] = []
    preview_rows: list[dict[str, Any]] = []
    block_reasons: list[str] = []
    preview_mode = "deterministic_preview"

    if plan.summary.selected_dataset_count == 0:
        block_reasons.append("No dataset was selected by the current assembly plan.")
    if plan.summary.selected_dataset_count > 1:
        block_reasons.append("Preview execution does not support multi-dataset joins yet.")

    pair_lookup = _pair_lookup(pool_mode=plan.pool_mode, cache_dir=cache_dir)
    selected_pair = pair_lookup.get(selected_dataset_ids[0]) if len(selected_dataset_ids) == 1 else None
    sample_rows = []
    if selected_pair is not None:
        sample_rows = list((getattr(selected_pair, "meta", {}) or {}).get("sample_rows") or [])
        if not sample_rows:
            block_reasons.append("Selected dataset does not expose deterministic sample rows for preview execution.")

    previewable_fields = 0
    for resolution in plan.field_resolutions:
        included = False
        notes = list(resolution.notes)
        if not block_reasons and sample_rows:
            ok, _value, note = _preview_value_from_resolution(resolution, sample_rows[0])
            included = ok
            if note:
                notes.append(note)
        manifest_fields.append(
            CompositionManifestField(
                target_field=resolution.target_field,
                resolution_status=resolution.resolution_status,
                source_tier=resolution.source_tier,
                source_dataset_id=resolution.source_dataset_id,
                source_columns=resolution.source_columns or ([resolution.source_column] if resolution.source_column else []),
                included_in_preview=included,
                notes=notes,
            )
        )
        if included:
            previewable_fields += 1

    if not block_reasons and sample_rows:
        for sample in sample_rows[:preview_row_limit]:
            rendered: dict[str, Any] = {}
            for resolution in plan.field_resolutions:
                ok, value, _note = _preview_value_from_resolution(resolution, sample)
                if ok:
                    rendered[resolution.target_field] = value
            if rendered:
                preview_rows.append(rendered)

    if previewable_fields == 0 and not block_reasons:
        block_reasons.append("No resolved fields were executable in deterministic preview mode.")

    manifest = CompositionManifest(
        selected_dataset_ids=selected_dataset_ids,
        selected_dataset_count=len(selected_dataset_ids),
        preview_mode=preview_mode,
        blocked=bool(block_reasons),
        block_reasons=block_reasons,
        fields=manifest_fields,
    )

    message = "Dataset composition preview generated." if preview_rows else "Dataset composition preview is blocked or empty under current deterministic constraints."
    return DatasetCompositionPreviewResponse(
        status="ok",
        message=message,
        assembly_plan=plan.model_dump(),
        manifest=manifest,
        preview_rows=preview_rows,
    )
