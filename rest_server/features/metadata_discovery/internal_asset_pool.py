from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from rest_server.features.metadata_discovery.column_profile import profile_csv_file, profile_tabular_rows

MANIFEST_VERSION = 1
INTERNAL_POOL_FILENAME = "internal_asset_schema_pool.json"


@dataclass
class InternalAssetSchemaPair:
    dataset_id: str
    title: str
    source_family: str
    source_url: str
    public_access: str
    schema: dict[str, Any]
    raw_schema: dict[str, Any]
    task_tags: dict[str, Any]
    provenance: dict[str, Any]
    meta: dict[str, Any]

    def to_matcher_record(self) -> dict[str, Any]:
        return {
            "id": self.dataset_id,
            "schema": self.schema,
            "meta": {
                "title": self.title,
                "source_family": self.source_family,
                "source_url": self.source_url,
                "public_access": self.public_access,
                "task_tags": self.task_tags,
                **self.meta,
            },
        }

    def to_manifest_entry(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_manifest_entry(cls, payload: dict[str, Any]) -> "InternalAssetSchemaPair":
        return cls(**payload)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _workspace_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "PATRA").is_dir():
            return candidate
    return current.parents[-1]


def _default_export_path() -> Path | None:
    override = (os.getenv("PATRA_DB_EXPORT_CSV") or "").strip()
    if override:
        path = Path(override)
        return path if path.is_file() else None

    export_dir = _workspace_root() / "PATRA" / "tmp" / "db_exports"
    if not export_dir.is_dir():
        return None
    candidates = sorted(export_dir.glob("patradb-public-export-*.csv"))
    return candidates[-1] if candidates else None


def _default_cache_dir() -> Path:
    return _workspace_root() / "PATRA" / ".patra-agent-cache"


def _normalize_cache_dir(cache_dir: str | None) -> Path:
    target = Path(cache_dir) if cache_dir else _default_cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    return target


def internal_pool_cache_path(cache_dir: str | None = None) -> Path:
    return _normalize_cache_dir(cache_dir) / INTERNAL_POOL_FILENAME


def _load_export_tables(path: Path) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            table_name = row["table_name"]
            payload = row["rows"] or "[]"
            tables[table_name] = json.loads(payload)
    return tables


def _schema_property(
    json_type: str,
    description: str,
    *,
    enum: list[str] | None = None,
    items_type: str | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prop: dict[str, Any] = {"type": json_type, "description": description}
    if enum:
        prop["enum"] = enum
    if json_type == "array":
        prop["items"] = {"type": items_type or "string"}
    if profile:
        prop["x-profile"] = profile
    return prop


def _build_schema(title: str, description: str, properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "description": f"{title}. {description}",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def _datasheet_lookup(rows: dict[str, list[dict[str, Any]]], table_name: str, field_name: str) -> dict[int, list[Any]]:
    output: dict[int, list[Any]] = {}
    for row in rows.get(table_name, []):
        datasheet_id = int(row["datasheet_id"])
        output.setdefault(datasheet_id, []).append(row.get(field_name))
    return output


def _pair_quality_tags(*, column_aware: bool, review_status: str | None = None) -> dict[str, Any]:
    return {
        "column_aware": column_aware,
        "metadata_only": not column_aware,
        "review_status": review_status,
    }


def _build_model_card_pairs(rows: dict[str, list[dict[str, Any]]], export_path: Path) -> list[InternalAssetSchemaPair]:
    pairs: list[InternalAssetSchemaPair] = []
    for row in rows.get("model_cards", []):
        title = row.get("name") or f"Model Card {row['id']}"
        short_description = row.get("short_description") or row.get("full_description") or "PATRA internal model card."
        schema = _build_schema(
            title,
            short_description,
            {
                "model_name": _schema_property("string", "PATRA model card title or record name."),
                "author": _schema_property("string", "Author or owning organization."),
                "category": _schema_property("string", "Declared model category or task family."),
                "keywords": _schema_property("string", "Comma-separated PATRA keywords."),
                "input_type": _schema_property("string", "Declared input modality or data type."),
                "input_data": _schema_property("string", "Source dataset or input reference."),
                "output_data": _schema_property("string", "Output artifact or model location."),
                "short_description": _schema_property("string", "Short PATRA summary."),
                "full_description": _schema_property("string", "Detailed PATRA narrative description."),
                "foundational_model": _schema_property("string", "Underlying foundation model when declared."),
            },
        )
        pairs.append(
            InternalAssetSchemaPair(
                dataset_id=f"patra_model_card_{row['id']}",
                title=title,
                source_family="patra_internal",
                source_url=f"/explore-model-cards/{row['id']}",
                public_access="private" if row.get("is_private") else "public",
                schema=schema,
                raw_schema=schema,
                task_tags={
                    "asset_type": "model_card",
                    "source_tier": "tier1_real",
                    "approved": row.get("status") == "approved",
                    "synthetic": False,
                    **_pair_quality_tags(column_aware=False, review_status=row.get("status")),
                },
                provenance={
                    "method": "patra_db_export_model_card",
                    "export_file": str(export_path),
                    "asset_id": row["id"],
                },
                meta={
                    "asset_type": "model_card",
                    "asset_id": row["id"],
                    "status": row.get("status"),
                    "author": row.get("author"),
                    "category": row.get("category"),
                },
            )
        )
    return pairs


def _build_datasheet_pairs(rows: dict[str, list[dict[str, Any]]], export_path: Path) -> list[InternalAssetSchemaPair]:
    title_lookup = _datasheet_lookup(rows, "datasheet_titles", "title")
    description_lookup = _datasheet_lookup(rows, "datasheet_descriptions", "description")
    subject_lookup = _datasheet_lookup(rows, "datasheet_subjects", "subject")
    pairs: list[InternalAssetSchemaPair] = []
    for row in rows.get("datasheets", []):
        identifier = int(row["identifier"])
        titles = [item for item in title_lookup.get(identifier, []) if item]
        descriptions = [item for item in description_lookup.get(identifier, []) if item]
        subjects = [item for item in subject_lookup.get(identifier, []) if item]
        title = titles[0] if titles else f"Datasheet {identifier}"
        description = descriptions[0] if descriptions else "PATRA internal datasheet."
        schema = _build_schema(
            title,
            description,
            {
                "datasheet_title": _schema_property("string", "Primary datasheet title."),
                "description": _schema_property("string", "Primary datasheet description."),
                "subjects": _schema_property("array", "Declared subject tags.", items_type="string"),
                "resource_type": _schema_property("string", "Datasheet resource type."),
                "resource_type_general": _schema_property("string", "Generalized resource type."),
                "format": _schema_property("string", "Declared file or resource format."),
                "size": _schema_property("string", "Declared dataset size."),
                "publication_year": _schema_property("integer", "Publication year."),
                "version": _schema_property("string", "Declared datasheet version."),
                "status": _schema_property("string", "Workflow status."),
            },
        )
        pairs.append(
            InternalAssetSchemaPair(
                dataset_id=f"patra_datasheet_{identifier}",
                title=title,
                source_family="patra_internal",
                source_url=f"/explore-datasheets/{identifier}",
                public_access="private" if row.get("is_private") else "public",
                schema=schema,
                raw_schema=schema,
                task_tags={
                    "asset_type": "datasheet",
                    "source_tier": "tier1_real",
                    "approved": row.get("status") == "approved",
                    "synthetic": False,
                    **_pair_quality_tags(column_aware=False, review_status=row.get("status")),
                },
                provenance={
                    "method": "patra_db_export_datasheet",
                    "export_file": str(export_path),
                    "asset_id": identifier,
                },
                meta={
                    "asset_type": "datasheet",
                    "asset_id": identifier,
                    "status": row.get("status"),
                    "subjects": subjects,
                },
            )
        )
    return pairs


def _profile_ingestion_artifact(row: dict[str, Any]) -> dict[str, Any] | None:
    staged_csv_path_text = (row.get("staged_csv_path") or "").strip()
    if staged_csv_path_text:
        staged_csv_path = Path(staged_csv_path_text)
        if staged_csv_path.is_file():
            profiled = profile_csv_file(staged_csv_path)
            profiled["profile_source"] = "staged_csv_path"
            return profiled

    headers = row.get("headers_sample") or []
    sample_rows = row.get("rows_sample") or []
    if headers:
        profiled = profile_tabular_rows(
            headers,
            [sample for sample in sample_rows if isinstance(sample, dict)],
            source_label=f"artifact:{row.get('id')}",
        )
        profiled["profile_source"] = "headers_sample_rows_sample"
        return profiled
    return None


def _build_ingestion_pairs(rows: dict[str, list[dict[str, Any]]], export_path: Path) -> list[InternalAssetSchemaPair]:
    pairs: list[InternalAssetSchemaPair] = []
    for row in rows.get("automated_ingestion_artifacts", []):
        profiled = _profile_ingestion_artifact(row)
        if not profiled or not profiled.get("schema_properties"):
            continue
        title = row.get("page_title") or f"Automated ingestion artifact {row['id']}"
        description = (
            row.get("datasheet_draft", {}).get("description")
            or f"Discovered CSV artifact from {row.get('source_url') or 'external source'}."
        )
        schema = _build_schema(title, description, profiled["schema_properties"])
        schema["required"] = profiled.get("required", [])
        pairs.append(
            InternalAssetSchemaPair(
                dataset_id=f"patra_ingestion_artifact_{row['id']}",
                title=title,
                source_family="patra_internal_ingestion",
                source_url=row.get("csv_url") or row.get("source_url") or "",
                public_access="private",
                schema=schema,
                raw_schema=schema,
                task_tags={
                    "asset_type": "automated_ingestion_artifact",
                    "source_tier": "tier1_real",
                    "approved": row.get("status") == "approved",
                    "synthetic": False,
                    **_pair_quality_tags(column_aware=True, review_status=row.get("status")),
                },
                provenance={
                    "method": "patra_db_export_automated_ingestion_artifact",
                    "export_file": str(export_path),
                    "artifact_id": row["id"],
                },
                meta={
                    "asset_type": "automated_ingestion_artifact",
                    "artifact_id": row["id"],
                    "status": row.get("status"),
                    "source_url": row.get("source_url"),
                    "header_count": profiled.get("column_count", 0),
                    "sample_row_count": profiled.get("sample_row_count", 0),
                    "profile_source": profiled.get("profile_source"),
                    "column_profiles": profiled.get("column_profiles", []),
                    "sample_headers": profiled.get("headers", []),
                    "sample_rows": [sample for sample in (row.get("rows_sample") or []) if isinstance(sample, dict)],
                },
            )
        )
    return pairs


def _build_pairs_from_export(export_path: Path) -> list[InternalAssetSchemaPair]:
    rows = _load_export_tables(export_path)
    pairs: list[InternalAssetSchemaPair] = []
    pairs.extend(_build_model_card_pairs(rows, export_path))
    pairs.extend(_build_datasheet_pairs(rows, export_path))
    pairs.extend(_build_ingestion_pairs(rows, export_path))
    pairs.sort(
        key=lambda pair: (
            0 if pair.task_tags.get("column_aware") else 1,
            0 if pair.task_tags.get("approved") else 1,
            pair.title.lower(),
        )
    )
    return pairs


def _manifest_from_pairs(export_path: Path, pairs: list[InternalAssetSchemaPair]) -> dict[str, Any]:
    counts_by_asset_type: dict[str, int] = {}
    counts_by_source_family: dict[str, int] = {}
    counts_by_quality: dict[str, int] = {"column_aware": 0, "metadata_only": 0}
    for pair in pairs:
        asset_type = str(pair.task_tags.get("asset_type") or "unknown")
        source_family = pair.source_family
        counts_by_asset_type[asset_type] = counts_by_asset_type.get(asset_type, 0) + 1
        counts_by_source_family[source_family] = counts_by_source_family.get(source_family, 0) + 1
        quality_key = "column_aware" if pair.task_tags.get("column_aware") else "metadata_only"
        counts_by_quality[quality_key] = counts_by_quality.get(quality_key, 0) + 1

    return {
        "version": MANIFEST_VERSION,
        "generated_at": _utc_now_iso(),
        "export_file": str(export_path),
        "export_last_modified": datetime.fromtimestamp(export_path.stat().st_mtime, UTC).isoformat(),
        "pair_count": len(pairs),
        "counts_by_asset_type": counts_by_asset_type,
        "counts_by_source_family": counts_by_source_family,
        "counts_by_quality": counts_by_quality,
        "records": [pair.to_manifest_entry() for pair in pairs],
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _build_internal_asset_schema_pool_cached(export_path_str: str) -> tuple[InternalAssetSchemaPair, ...]:
    export_path = Path(export_path_str)
    return tuple(_build_pairs_from_export(export_path))


def build_internal_asset_schema_pool(export_path: str | None = None) -> list[InternalAssetSchemaPair]:
    resolved = Path(export_path) if export_path else _default_export_path()
    if resolved is None or not resolved.is_file():
        return []
    return list(_build_internal_asset_schema_pool_cached(str(resolved.resolve())))


def materialize_internal_asset_schema_pool(cache_dir: str | None = None, export_path: str | None = None) -> dict[str, Any]:
    resolved_export = Path(export_path) if export_path else _default_export_path()
    if resolved_export is None or not resolved_export.is_file():
        raise FileNotFoundError("No PATRA database export CSV is available for internal asset indexing.")

    pairs = build_internal_asset_schema_pool(str(resolved_export))
    manifest = _manifest_from_pairs(resolved_export, pairs)
    cache_path = internal_pool_cache_path(cache_dir)
    cache_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_materialized_internal_asset_schema_pool(
    cache_dir: str | None = None,
    export_path: str | None = None,
    *,
    rebuild_if_missing: bool = True,
) -> list[InternalAssetSchemaPair]:
    cache_path = internal_pool_cache_path(cache_dir)
    if not cache_path.is_file():
        if not rebuild_if_missing:
            return []
        materialize_internal_asset_schema_pool(cache_dir=cache_dir, export_path=export_path)
    manifest = _load_manifest(cache_path)
    return [InternalAssetSchemaPair.from_manifest_entry(item) for item in manifest.get("records", [])]


def get_internal_asset_index_status(cache_dir: str | None = None, export_path: str | None = None) -> dict[str, Any]:
    resolved_export = Path(export_path) if export_path else _default_export_path()
    cache_path = internal_pool_cache_path(cache_dir)

    export_exists = bool(resolved_export and resolved_export.is_file())
    cache_exists = cache_path.is_file()

    status: dict[str, Any] = {
        "status": "missing",
        "export_file": str(resolved_export) if resolved_export else None,
        "export_exists": export_exists,
        "cache_file": str(cache_path),
        "cache_exists": cache_exists,
        "pair_count": 0,
        "counts_by_asset_type": {},
        "counts_by_source_family": {},
        "counts_by_quality": {},
        "generated_at": None,
    }

    if not export_exists:
        status["status"] = "missing_export"
        return status

    if not cache_exists:
        status["status"] = "needs_materialization"
        return status

    manifest = _load_manifest(cache_path)
    status["status"] = "ready"
    status["pair_count"] = manifest.get("pair_count", 0)
    status["counts_by_asset_type"] = manifest.get("counts_by_asset_type", {})
    status["counts_by_source_family"] = manifest.get("counts_by_source_family", {})
    status["counts_by_quality"] = manifest.get("counts_by_quality", {})
    status["generated_at"] = manifest.get("generated_at")
    status["export_last_modified"] = manifest.get("export_last_modified")
    return status
