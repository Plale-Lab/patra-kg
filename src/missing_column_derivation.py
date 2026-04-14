"""
Deterministic missing-column feasibility for PATRA substitution workflows.

PATRA is domain-agnostic (model cards, datasets, reproducibility across topics).
This module implements the **V1 catalog** of auditable, code-defined rules that
decide whether a **target attribute** in a query schema can be satisfied from a
candidate dataset's normalized + raw schemas.

- Any property name may appear in the query schema; only attributes matched by an
  explicit rule are marked ``derivable with provenance``.
- Everything else is ``not safely derivable`` (default deny) until a new rule is
  added—see ``PATRA/patra-backend-src/docs/V1_DETERMINISTIC_DERIVATION_BOUNDARY.md``.

To extend PATRA for a new domain, add predicates + decisions here and wire
corresponding execution in ``patra_synthesis_service`` (backend).
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class ColumnDerivationDecision:
    target_field: str
    status: str
    rationale: str
    source_fields: List[str]
    checks: List[str]


def _properties(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    props = schema.get("properties", {})
    return props if isinstance(props, dict) else {}


def _lower_fields(schema: Dict[str, Any]) -> Dict[str, str]:
    return {name.lower(): name for name in _properties(schema)}


def _find_any(field_lookup: Dict[str, str], *needles: str) -> List[str]:
    found: List[str] = []
    for lowered, original in field_lookup.items():
        if any(needle in lowered for needle in needles):
            found.append(original)
    return sorted(set(found))


def _has_date_support(field_lookup: Dict[str, str]) -> List[str]:
    return _find_any(field_lookup, "date", "time", "year")


def _direct_or_derivable(
    target_field: str,
    candidate_schema: Dict[str, Any],
    raw_schema: Dict[str, Any],
) -> ColumnDerivationDecision:
    candidate_props = _properties(candidate_schema)
    raw_lookup = _lower_fields(raw_schema)

    if target_field in candidate_props:
        return ColumnDerivationDecision(
            target_field=target_field,
            status="directly available",
            rationale="The candidate normalized schema already exposes this target field group.",
            source_fields=[target_field],
            checks=["normalized schema presence"],
        )

    if target_field == "LAT":
        sources = _find_any(raw_lookup, "latitude", "lat")
        if sources:
            return ColumnDerivationDecision(
                target_field=target_field,
                status="derivable with provenance",
                rationale="Latitude can be normalized from an existing latitude field.",
                source_fields=sources,
                checks=["coordinate alias normalization"],
            )

    if target_field == "LON":
        sources = _find_any(raw_lookup, "longitude", "long", "lon")
        if sources:
            return ColumnDerivationDecision(
                target_field=target_field,
                status="derivable with provenance",
                rationale="Longitude can be normalized from an existing longitude field.",
                source_fields=sources,
                checks=["coordinate alias normalization"],
            )

    if target_field == "Year":
        year_fields = _find_any(raw_lookup, "year")
        date_fields = _find_any(raw_lookup, "date")
        if year_fields:
            return ColumnDerivationDecision(
                target_field=target_field,
                status="derivable with provenance",
                rationale="A year value can be normalized directly from an existing year field.",
                source_fields=year_fields,
                checks=["year alias normalization"],
            )
        if date_fields:
            return ColumnDerivationDecision(
                target_field=target_field,
                status="derivable with provenance",
                rationale="A year value can be extracted deterministically from an existing date field.",
                source_fields=date_fields,
                checks=["date parsing", "calendar-year extraction"],
            )

    if target_field == "yield":
        sources = _find_any(raw_lookup, "yield", "yield_kg", "real_tch")
        if sources:
            return ColumnDerivationDecision(
                target_field=target_field,
                status="derivable with provenance",
                rationale="The target yield can be normalized from an existing yield-like field.",
                source_fields=sources,
                checks=["yield alias normalization", "unit verification"],
            )
        value_fields = _find_any(raw_lookup, "value")
        element_fields = _find_any(raw_lookup, "element")
        if value_fields and element_fields:
            return ColumnDerivationDecision(
                target_field=target_field,
                status="derivable with provenance",
                rationale="A yield column can be filtered from a long-form value table when the element dimension explicitly identifies yield.",
                source_fields=sorted(set(value_fields + element_fields)),
                checks=["element filter", "yield unit verification"],
            )

    if target_field == "Tmax_monthly":
        sources = _find_any(raw_lookup, "tmax")
        dates = _has_date_support(raw_lookup)
        if sources and dates:
            return ColumnDerivationDecision(
                target_field=target_field,
                status="derivable with provenance",
                rationale="Monthly maximum temperature can be aggregated from dated Tmax observations.",
                source_fields=sources + dates,
                checks=["date parsing", "monthly aggregation", "temperature unit preservation"],
            )

    if target_field == "Tmin_monthly":
        sources = _find_any(raw_lookup, "tmin")
        dates = _has_date_support(raw_lookup)
        if sources and dates:
            return ColumnDerivationDecision(
                target_field=target_field,
                status="derivable with provenance",
                rationale="Monthly minimum temperature can be aggregated from dated Tmin observations.",
                source_fields=sources + dates,
                checks=["date parsing", "monthly aggregation", "temperature unit preservation"],
            )

    if target_field == "PRE_monthly":
        sources = _find_any(raw_lookup, "precip", "rain", " pr")
        sources += [name for name in raw_lookup.values() if name.lower() == "pr"]
        dates = _has_date_support(raw_lookup)
        if sources and dates:
            return ColumnDerivationDecision(
                target_field=target_field,
                status="derivable with provenance",
                rationale="Monthly precipitation can be aggregated from dated precipitation observations.",
                source_fields=sorted(set(sources + dates)),
                checks=["date parsing", "monthly aggregation", "precipitation unit verification"],
            )

    if target_field == "NDVI_monthly":
        sources = _find_any(raw_lookup, "ndvi")
        dates = _has_date_support(raw_lookup)
        if sources and dates:
            return ColumnDerivationDecision(
                target_field=target_field,
                status="derivable with provenance",
                rationale="Monthly NDVI can be aggregated from dated NDVI observations.",
                source_fields=sources + dates,
                checks=["date parsing", "monthly aggregation", "index range validation"],
            )

    if target_field == "SM_monthly":
        sources = _find_any(raw_lookup, "soil_moisture", "soil moisture", "sm")
        dates = _has_date_support(raw_lookup)
        if sources and dates:
            return ColumnDerivationDecision(
                target_field=target_field,
                status="derivable with provenance",
                rationale="Monthly soil moisture can be aggregated from dated soil-moisture observations.",
                source_fields=sources + dates,
                checks=["date parsing", "monthly aggregation", "soil-moisture range validation"],
            )

    return ColumnDerivationDecision(
        target_field=target_field,
        status="not safely derivable",
        rationale="No strict deterministic derivation rule is available for this target attribute under the current V1 boundary.",
        source_fields=[],
        checks=[],
    )


def analyze_missing_columns(
    query_schema: Dict[str, Any],
    candidate_schema: Dict[str, Any],
    raw_schema: Dict[str, Any],
) -> List[ColumnDerivationDecision]:
    return [
        _direct_or_derivable(target_field, candidate_schema, raw_schema)
        for target_field in _properties(query_schema)
    ]


def build_derivation_summary(
    decisions: Iterable[ColumnDerivationDecision],
) -> Dict[str, Any]:
    rows = list(decisions)
    return {
        "direct_count": sum(1 for row in rows if row.status == "directly available"),
        "derivable_count": sum(1 for row in rows if row.status == "derivable with provenance"),
        "rejected_count": sum(1 for row in rows if row.status == "not safely derivable"),
        "rows": [asdict(row) for row in rows],
    }
