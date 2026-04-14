from __future__ import annotations

import logging
import re

from rest_server.features.metadata_discovery.internal_asset_pool import (
    get_internal_asset_index_status,
    load_materialized_internal_asset_schema_pool,
    materialize_internal_asset_schema_pool,
)
from rest_server.features.intent_schema.models import IntentSchemaField, IntentSchemaResult
from rest_server.features.metadata_discovery.models import MetadataDiscoveryCoverage
from rest_server.patra_agent_service import rank_query_schema_candidates, rank_query_schema_candidates_from_pairs


_ENUM_PREFIX_RE = re.compile(r"^enum:\s*", re.IGNORECASE)
_NUMERIC_RANGE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")
_GE_RANGE_RE = re.compile(r"^\s*>=\s*(-?\d+(?:\.\d+)?)\s*$")
log = logging.getLogger(__name__)


def _schema_type_and_format(field: IntentSchemaField) -> tuple[str, str | None]:
    mapping = {
        "integer": ("integer", None),
        "float": ("number", None),
        "boolean": ("boolean", None),
        "date": ("string", "date"),
        "datetime": ("string", "date-time"),
        "string": ("string", None),
    }
    return mapping.get(field.data_type, ("string", None))


def _apply_expected_range(prop: dict, field: IntentSchemaField) -> None:
    text = (field.expected_range or "").strip()
    if not text:
        return

    schema_type = prop.get("type")
    if schema_type in {"string", "boolean"}:
        enum_text = _ENUM_PREFIX_RE.sub("", text)
        if text.lower().startswith("enum:"):
            values = [item.strip() for item in enum_text.split("|") if item.strip()]
            if values:
                prop["enum"] = values
            return
        if schema_type == "boolean":
            prop["enum"] = [True, False]
            return
        return

    match = _NUMERIC_RANGE_RE.match(text)
    if match:
        minimum, maximum = match.groups()
        prop["minimum"] = float(minimum)
        prop["maximum"] = float(maximum)
        return

    match = _GE_RANGE_RE.match(text)
    if match:
        prop["minimum"] = float(match.group(1))


def _field_property(field: IntentSchemaField) -> dict:
    schema_type, schema_format = _schema_type_and_format(field)
    prop = {
        "type": schema_type,
        "description": field.description,
        "x-patra-source-name": field.name,
        "x-patra-canonical-name": field.name,
        "x-patra-confidence": "high",
        "x-patra-role": field.semantic_role,
    }
    if schema_format:
        prop["format"] = schema_format
    if field.notes:
        prop["x-patra-notes"] = field.notes
    if field.distribution_expectation:
        prop["x-profile"] = {"distribution": field.distribution_expectation}
    _apply_expected_range(prop, field)
    return prop


def intent_schema_to_query_schema(intent_schema: IntentSchemaResult) -> dict:
    required: list[str] = []
    properties: dict[str, dict] = {}

    for field in intent_schema.schema_fields:
        properties[field.name] = _field_property(field)
        if field.required:
            required.append(field.name)

    return {
        "title": intent_schema.intent_summary,
        "type": "object",
        "required": required,
        "properties": properties,
        "x-patra-entity-grain": intent_schema.entity_grain,
        "x-patra-task-type": intent_schema.task_type,
        "x-patra-target-column": intent_schema.target_column,
        "x-patra-label-definition": intent_schema.label_definition,
        "x-patra-prediction-horizon": intent_schema.prediction_horizon,
    }


def _winner_coverage(response: dict, total_fields: int) -> MetadataDiscoveryCoverage | None:
    ranking = response.get("ranking") or []
    if not ranking:
        return None
    winner = ranking[0]
    return MetadataDiscoveryCoverage(
        direct_count=len(winner.get("matched_field_groups") or []),
        derivable_count=len(winner.get("derivable_field_groups") or []),
        missing_count=len(winner.get("missing_field_groups") or []),
        total_fields=total_fields,
    )


def discover_metadata(
    *,
    intent_schema: IntentSchemaResult,
    top_k: int,
    disable_llm: bool,
    api_base: str | None,
    model: str | None,
    api_key: str | None,
    timeout_seconds: int,
    cache_dir: str | None,
) -> dict:
    query_schema = intent_schema_to_query_schema(intent_schema)
    internal_pairs = load_materialized_internal_asset_schema_pool(cache_dir=cache_dir, rebuild_if_missing=True)
    try:
        if internal_pairs:
            response = rank_query_schema_candidates_from_pairs(
                query_schema,
                pairs=internal_pairs,
                top_k=top_k,
                disable_llm=disable_llm,
                api_base=api_base,
                model=model,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                success_message="Intent schema ranked against the PATRA internal asset pool using the shared agent matching stack.",
            )
        else:
            response = rank_query_schema_candidates(
                query_schema,
                top_k=top_k,
                disable_llm=disable_llm,
                api_base=api_base,
                model=model,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                cache_dir=cache_dir,
                success_message="No internal asset export was available; attempted public fallback schema pool.",
            )
    except Exception as exc:  # noqa: BLE001
        log.exception("metadata_discovery.discover_metadata.ranking_failed")
        response = {
            "status": "degraded",
            "message": f"Metadata discovery could not rank candidate datasets: {exc}",
            "query_schema": query_schema,
            "candidate_count": 0,
            "winner_dataset_id": None,
            "ranking": [],
        }
    coverage = _winner_coverage(response, len(intent_schema.schema_fields))
    response["winner_coverage"] = coverage.model_dump() if coverage is not None else None
    response["pool_mode"] = "internal" if internal_pairs else "public"
    return response


def internal_asset_index_status(*, cache_dir: str | None) -> dict:
    return get_internal_asset_index_status(cache_dir=cache_dir)


def refresh_internal_asset_index(*, cache_dir: str | None) -> dict:
    manifest = materialize_internal_asset_schema_pool(cache_dir=cache_dir)
    return {
        "status": "ok",
        "message": "Internal PATRA asset schema pool materialized.",
        "export_file": manifest["export_file"],
        "cache_file": str(get_internal_asset_index_status(cache_dir=cache_dir)["cache_file"]),
        "pair_count": manifest.get("pair_count", 0),
        "counts_by_asset_type": manifest.get("counts_by_asset_type", {}),
        "counts_by_source_family": manifest.get("counts_by_source_family", {}),
        "counts_by_quality": manifest.get("counts_by_quality", {}),
        "generated_at": manifest.get("generated_at"),
    }
