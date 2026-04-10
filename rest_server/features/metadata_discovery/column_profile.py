from __future__ import annotations

import csv
import math
import re
from collections import Counter
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any


MAX_ENUM_VALUES = 12
MAX_TOP_VALUES = 8
DEFAULT_SAMPLE_LIMIT = 250


def _normalize_header(value: Any) -> str:
    return str(value or "").strip()


def _canonical_field_name(value: Any) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if normalized and normalized[0].isdigit():
        normalized = f"field_{normalized}"
    return normalized or "field"


def _normalize_cell(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    cleaned = str(value).strip()
    return cleaned or None


def _try_bool(value: str) -> bool | None:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"yes", "no"}:
        return lowered == "yes"
    if lowered in {"1", "0"}:
        return lowered == "1"
    return None


def _try_int(value: str) -> int | None:
    try:
        if value.startswith("0") and len(value) > 1 and not value.startswith("0."):
            return None
        return int(value)
    except ValueError:
        return None


def _try_float(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _try_datetime(value: str) -> datetime | None:
    candidate = value.strip()
    if not candidate:
        return None
    normalized = candidate.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue
    return None


def _try_date(value: str) -> date | None:
    candidate = value.strip()
    if not candidate:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue
    return None


def _infer_scalar_kind(value: str) -> tuple[str, Any]:
    parsed_bool = _try_bool(value)
    if parsed_bool is not None:
        return "boolean", parsed_bool

    parsed_int = _try_int(value)
    if parsed_int is not None:
        return "integer", parsed_int

    parsed_float = _try_float(value)
    if parsed_float is not None:
        return "number", parsed_float

    parsed_datetime = _try_datetime(value)
    if parsed_datetime is not None:
        return "datetime", parsed_datetime

    parsed_date = _try_date(value)
    if parsed_date is not None:
        return "date", parsed_date

    return "string", value


def _canonical_type(kinds: list[str]) -> tuple[str, str | None]:
    if not kinds:
        return "string", None
    unique = set(kinds)
    if unique == {"boolean"}:
        return "boolean", None
    if unique <= {"integer"}:
        return "integer", None
    if unique <= {"integer", "number"}:
        return "number", None
    if unique == {"date"}:
        return "date", "date"
    if unique <= {"date", "datetime"}:
        return "datetime", "date-time"
    return "string", None


def _json_schema_type(canonical_type: str) -> str:
    mapping = {
        "boolean": "boolean",
        "integer": "integer",
        "number": "number",
        "date": "string",
        "datetime": "string",
        "string": "string",
    }
    return mapping.get(canonical_type, "string")


def _coerce_json(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _enum_values_for_column(canonical_type: str, values: list[str], top_values: list[str]) -> list[Any] | None:
    if canonical_type not in {"string", "boolean"}:
        return None
    unique = sorted(set(values))
    if not unique or len(unique) > MAX_ENUM_VALUES:
        return None
    if canonical_type == "string":
        avg_len = sum(len(item) for item in unique) / len(unique)
        if avg_len > 40:
            return None
        return unique
    return [item.lower() in {"true", "yes", "1"} for item in unique]


def _build_column_profile(name: str, values: list[str | None]) -> dict[str, Any]:
    normalized_values = [_normalize_cell(value) for value in values]
    non_null_values = [value for value in normalized_values if value is not None]
    inferred_kinds: list[str] = []
    parsed_values: list[Any] = []
    for value in non_null_values:
        kind, parsed = _infer_scalar_kind(value)
        inferred_kinds.append(kind)
        parsed_values.append(parsed)

    canonical_type, inferred_format = _canonical_type(inferred_kinds)
    counter = Counter(non_null_values)
    top_values = [item for item, _count in counter.most_common(MAX_TOP_VALUES)]

    profile: dict[str, Any] = {
        "name": name,
        "canonical_name": _canonical_field_name(name),
        "canonical_type": canonical_type,
        "json_schema_type": _json_schema_type(canonical_type),
        "inferred_format": inferred_format,
        "description": f"Observed column from deterministic CSV profiling: {name}.",
        "sample_size": len(values),
        "non_null_count": len(non_null_values),
        "null_count": len(values) - len(non_null_values),
        "null_ratio": round((len(values) - len(non_null_values)) / len(values), 4) if values else 0.0,
        "unique_count": len(counter),
        "top_values": top_values,
        "nullable": len(non_null_values) != len(values),
    }

    enum_values = _enum_values_for_column(canonical_type, non_null_values, top_values)
    if enum_values:
        profile["enum_values"] = enum_values

    if canonical_type in {"integer", "number"} and parsed_values:
        numeric_values = [float(value) for value in parsed_values]
        profile["min_value"] = min(numeric_values)
        profile["max_value"] = max(numeric_values)
    elif canonical_type in {"date", "datetime"} and parsed_values:
        profile["min_value"] = _coerce_json(min(parsed_values))
        profile["max_value"] = _coerce_json(max(parsed_values))

    return profile


def _schema_property_from_profile(profile: dict[str, Any]) -> dict[str, Any]:
    prop: dict[str, Any] = {
        "type": profile["json_schema_type"],
        "description": profile["description"],
        "x-patra-source-name": profile["name"],
        "x-patra-canonical-name": profile["canonical_name"],
        "x-profile": {
            "sample_size": profile["sample_size"],
            "non_null_count": profile["non_null_count"],
            "null_count": profile["null_count"],
            "null_ratio": profile["null_ratio"],
            "unique_count": profile["unique_count"],
            "top_values": profile["top_values"],
        },
        "x-patra-source-kind": "deterministic_column_profile",
        "x-patra-canonical-type": profile["canonical_type"],
    }
    inferred_format = profile.get("inferred_format")
    if inferred_format:
        prop["format"] = inferred_format
    enum_values = profile.get("enum_values")
    if enum_values:
        prop["enum"] = enum_values
    if "min_value" in profile and profile["canonical_type"] in {"integer", "number"}:
        prop["minimum"] = profile["min_value"]
    if "max_value" in profile and profile["canonical_type"] in {"integer", "number"}:
        prop["maximum"] = profile["max_value"]
    if "min_value" in profile and profile["canonical_type"] in {"date", "datetime"}:
        prop["x-profile"]["min_value"] = profile["min_value"]
    if "max_value" in profile and profile["canonical_type"] in {"date", "datetime"}:
        prop["x-profile"]["max_value"] = profile["max_value"]
    return prop


def profile_tabular_rows(
    headers: list[str],
    rows: list[dict[str, Any]],
    *,
    source_label: str,
) -> dict[str, Any]:
    cleaned_headers = [_normalize_header(header) for header in headers if _normalize_header(header)]
    column_profiles: list[dict[str, Any]] = []
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    seen_property_names: set[str] = set()

    for header in cleaned_headers:
        values = [row.get(header) if isinstance(row, dict) else None for row in rows]
        profile = _build_column_profile(header, values)
        property_name = profile["canonical_name"]
        if property_name in seen_property_names:
            suffix = 2
            while f"{property_name}_{suffix}" in seen_property_names:
                suffix += 1
            property_name = f"{property_name}_{suffix}"
            profile["canonical_name"] = property_name
        column_profiles.append(profile)
        properties[property_name] = _schema_property_from_profile(profile)
        seen_property_names.add(property_name)
        if not profile["nullable"]:
            required.append(property_name)

    return {
        "source_label": source_label,
        "headers": cleaned_headers,
        "sample_row_count": len(rows),
        "column_count": len(cleaned_headers),
        "column_profiles": column_profiles,
        "schema_properties": properties,
        "required": required,
    }


def profile_csv_text(
    csv_text: str,
    *,
    source_label: str,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> dict[str, Any]:
    reader = csv.DictReader(StringIO(csv_text))
    headers = [header for header in (reader.fieldnames or []) if _normalize_header(header)]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(reader):
        if index >= sample_limit:
            break
        rows.append({header: row.get(header) for header in headers})
    return profile_tabular_rows(headers, rows, source_label=source_label)


def profile_csv_bytes(
    payload: bytes,
    *,
    source_label: str,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    text = payload.decode(encoding, errors="replace")
    return profile_csv_text(text, source_label=source_label, sample_limit=sample_limit)


def profile_csv_file(
    path: Path,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    text = path.read_text(encoding=encoding, errors="replace")
    return profile_csv_text(text, source_label=str(path), sample_limit=sample_limit)
