import logging
import json
from collections.abc import Sequence
from typing import NamedTuple

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status

from rest_server.asset_backups import ensure_initial_backup, record_backup
from rest_server.database import get_pool
from rest_server.deps import AssetIngestPrincipal, get_request_actor, require_asset_ingest_principal
from rest_server.asset_create_models import (
    AssetBulkDatasheetCreate,
    AssetBulkIngestResult,
    AssetBulkItemResult,
    AssetBulkModelCardCreate,
    AssetDatasheetCreate,
    AssetIngestResult,
    AssetModelCardCreate,
    AssetUpdateResult,
)
from rest_server.models import AssetBackupRecord, AssetBackupRunResult, AssetChangeLogEntry, AssetFieldChange, EditableRecordSummary

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/assets", tags=["assets"])


class AssetRevisionContext(NamedTuple):
    asset_version: int
    previous_version_id: int | None
    root_version_id: int | None


def _normalize_text_value(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).strip()
    return text or None


def _normalize_joined(values: Sequence[str] | None) -> str | None:
    if not values:
        return None
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return ", ".join(cleaned) if cleaned else None


def _model_card_edit_state_from_snapshot(snapshot: dict) -> dict[str, str | None]:
    card = snapshot.get("card") or {}
    model = snapshot.get("model") or {}
    return {
        "name": _normalize_text_value(card.get("name")),
        "version": _normalize_text_value(card.get("version")),
        "short_description": _normalize_text_value(card.get("short_description")),
        "full_description": _normalize_text_value(card.get("full_description")),
        "category": _normalize_text_value(card.get("category")),
        "input_type": _normalize_text_value(card.get("input_type")),
        "framework": _normalize_text_value(model.get("framework")),
        "test_accuracy": _normalize_text_value(model.get("test_accuracy")),
        "input_data": _normalize_text_value(card.get("input_data")),
        "output_data": _normalize_text_value(card.get("output_data")),
        "license": _normalize_text_value(model.get("license")),
        "keywords": _normalize_text_value(card.get("keywords")),
        "is_private": _normalize_text_value(card.get("is_private")),
    }


def _model_card_edit_state_from_asset(asset: AssetModelCardCreate) -> dict[str, str | None]:
    ai_model = asset.ai_model
    return {
        "name": _normalize_text_value(asset.name),
        "version": _normalize_text_value(asset.version),
        "short_description": _normalize_text_value(asset.short_description),
        "full_description": _normalize_text_value(asset.full_description),
        "category": _normalize_text_value(asset.category),
        "input_type": _normalize_text_value(asset.input_type),
        "framework": _normalize_text_value(ai_model.framework if ai_model else None),
        "test_accuracy": _normalize_text_value(ai_model.test_accuracy if ai_model else None),
        "input_data": _normalize_text_value(asset.input_data),
        "output_data": _normalize_text_value(asset.output_data),
        "license": _normalize_text_value(ai_model.license if ai_model else None),
        "keywords": _normalize_text_value(asset.keywords),
        "is_private": _normalize_text_value(asset.is_private),
    }


def _datasheet_edit_state_from_snapshot(snapshot: dict) -> dict[str, str | None]:
    core = snapshot.get("core") or {}
    return {
        "name": _normalize_text_value((snapshot.get("titles") or [{}])[0].get("title") if snapshot.get("titles") else None),
        "version": _normalize_text_value(core.get("version")),
        "description": _normalize_text_value((snapshot.get("descriptions") or [{}])[0].get("description") if snapshot.get("descriptions") else None),
        "source": _normalize_text_value(core.get("resource_type")),
        "publication_year": _normalize_text_value(core.get("publication_year")),
        "publisher": _normalize_text_value(core.get("publisher_name")),
        "download_url": _normalize_text_value((snapshot.get("related_identifiers") or [{}])[0].get("related_identifier") if snapshot.get("related_identifiers") else None),
        "creator": _normalize_joined([row.get("creator_name") for row in snapshot.get("creators") or []]),
        "features": _normalize_joined([row.get("subject") for row in snapshot.get("subjects") or []]),
        "is_private": _normalize_text_value(core.get("is_private")),
    }


def _datasheet_edit_state_from_asset(asset: AssetDatasheetCreate) -> dict[str, str | None]:
    return {
        "name": _normalize_text_value(asset.titles[0].title if asset.titles else None),
        "version": _normalize_text_value(asset.version),
        "description": _normalize_text_value(asset.descriptions[0].description if asset.descriptions else None),
        "source": _normalize_text_value(asset.resource_type),
        "publication_year": _normalize_text_value(asset.publication_year),
        "publisher": _normalize_text_value(asset.publisher.name if asset.publisher else None),
        "download_url": _normalize_text_value(asset.related_identifiers[0].related_identifier if asset.related_identifiers else None),
        "creator": _normalize_joined([item.creator_name for item in asset.creators]),
        "features": _normalize_joined([item.subject for item in asset.subjects]),
        "is_private": _normalize_text_value(asset.is_private),
    }


def _build_field_changes(before_state: dict[str, str | None], after_state: dict[str, str | None]) -> list[dict[str, str | None]]:
    changes: list[dict[str, str | None]] = []
    for field in sorted(set(before_state.keys()) | set(after_state.keys())):
        before_value = before_state.get(field)
        after_value = after_state.get(field)
        if before_value == after_value:
            continue
        changes.append({
            "field": field,
            "before": before_value,
            "after": after_value,
            "statement": f"{field}: {repr(before_value)} -> {repr(after_value)}",
        })
    return changes


def _coerce_change_items(raw_changes) -> list[dict]:
    if raw_changes is None:
        return []
    if isinstance(raw_changes, str):
        try:
            parsed = json.loads(raw_changes)
        except json.JSONDecodeError:
            return []
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(raw_changes, list):
        return [item for item in raw_changes if isinstance(item, dict)]
    return []


async def _record_change_log(
    conn: asyncpg.Connection,
    asset_type: str,
    asset_id: int,
    asset_version: int,
    changed_by: str | None,
    changes: list[dict[str, str | None]],
) -> int | None:
    if not changes:
        return None
    summary = changes[0]["statement"] if len(changes) == 1 else f"{changes[0]['statement']} (+{len(changes) - 1} more)"
    change_log_id = await conn.fetchval(
        """
        INSERT INTO asset_change_logs (
            asset_type, asset_id, asset_version, changed_by, changed_at, changes, summary
        )
        VALUES ($1, $2, $3, $4, NOW(), $5::jsonb, $6)
        RETURNING id
        """,
        asset_type,
        asset_id,
        asset_version,
        changed_by,
        json.dumps(changes),
        summary,
    )
    return int(change_log_id)


def _model_card_revision_context(asset: AssetModelCardCreate) -> AssetRevisionContext | None:
    if asset.asset_version is None and asset.previous_version_id is None and asset.root_version_id is None:
        return None
    return AssetRevisionContext(
        asset_version=asset.asset_version or 1,
        previous_version_id=asset.previous_version_id,
        root_version_id=asset.root_version_id,
    )


def _datasheet_revision_context(asset: AssetDatasheetCreate) -> AssetRevisionContext | None:
    if asset.asset_version is None and asset.previous_version_id is None and asset.root_version_id is None:
        return None
    return AssetRevisionContext(
        asset_version=asset.asset_version or 1,
        previous_version_id=asset.previous_version_id,
        root_version_id=asset.root_version_id,
    )


async def _find_duplicate_model_card(conn: asyncpg.Connection, asset: AssetModelCardCreate) -> int | None:
    row = await conn.fetchrow(
        """
        SELECT id
        FROM model_cards
        WHERE name = $1
          AND COALESCE(version, '') = COALESCE($2, '')
          AND COALESCE(author, '') = COALESCE($3, '')
          AND COALESCE(short_description, '') = COALESCE($4, '')
        LIMIT 1
        """,
        asset.name,
        asset.version,
        asset.author,
        asset.short_description,
    )
    return int(row["id"]) if row else None


async def _create_model_card_in_tx(
    conn: asyncpg.Connection,
    asset: AssetModelCardCreate,
    organization: str,
    revision_context: AssetRevisionContext | None = None,
) -> AssetIngestResult:
    duplicate_id = await _find_duplicate_model_card(conn, asset)
    if duplicate_id is not None:
        return AssetIngestResult(
            asset_type="model_card",
            asset_id=duplicate_id,
            organization=organization,
            created=False,
            duplicate=True,
        )

    model_card_id = await conn.fetchval(
        """
        INSERT INTO model_cards (
            name, version, is_private, is_gated, status,
            short_description, full_description, keywords, author, citation,
            input_data, input_type, output_data, foundational_model, category, documentation,
            created_at, updated_at, asset_version, previous_version_id, root_version_id
        )
        VALUES (
            $1, $2, $3, $4, 'approved',
            $5, $6, $7, $8, $9,
            $10, $11, $12, $13, $14, $15,
            NOW(), NOW(), $16, $17, $18
        )
        RETURNING id
        """,
        asset.name,
        asset.version,
        asset.is_private,
        asset.is_gated,
        asset.short_description,
        asset.full_description,
        asset.keywords,
        asset.author,
        asset.citation,
        asset.input_data,
        asset.input_type,
        asset.output_data,
        asset.foundational_model,
        asset.category,
        asset.documentation,
        revision_context.asset_version if revision_context else 1,
        revision_context.previous_version_id if revision_context else None,
        revision_context.root_version_id if revision_context else None,
    )

    if revision_context is None:
        await conn.execute(
            """
            UPDATE model_cards
            SET root_version_id = $2
            WHERE id = $1
            """,
            model_card_id,
            model_card_id,
        )

    if asset.ai_model is not None:
        await conn.execute(
            """
            INSERT INTO models (
                name, version, description, owner, location, license, framework, model_type,
                test_accuracy, model_metrics, inference_labels, model_structure,
                created_at, updated_at, model_card_id
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8,
                $9, $10::jsonb, $11::jsonb, $12::jsonb,
                NOW(), NOW(), $13
            )
            """,
            asset.ai_model.name,
            asset.ai_model.version,
            asset.ai_model.description,
            asset.ai_model.owner,
            asset.ai_model.location,
            asset.ai_model.license,
            asset.ai_model.framework,
            asset.ai_model.model_type,
            asset.ai_model.test_accuracy,
            json.dumps(asset.ai_model.model_metrics),
            json.dumps(asset.ai_model.inference_labels),
            json.dumps(asset.ai_model.model_structure),
            model_card_id,
        )

    log.info("Asset ingest: created model card %s for org %s", model_card_id, organization)
    return AssetIngestResult(
        asset_type="model_card",
        asset_id=int(model_card_id),
        organization=organization,
        created=True,
    )


async def _find_publisher_id(conn: asyncpg.Connection, publisher: dict | None) -> int | None:
    if not publisher:
        return None
    existing_id = await conn.fetchval(
        """
        SELECT id
        FROM publishers
        WHERE name = $1
          AND COALESCE(publisher_identifier, '') = COALESCE($2, '')
          AND COALESCE(publisher_identifier_scheme, '') = COALESCE($3, '')
          AND COALESCE(scheme_uri, '') = COALESCE($4, '')
          AND COALESCE(lang, '') = COALESCE($5, '')
        LIMIT 1
        """,
        publisher["name"],
        publisher.get("publisher_identifier"),
        publisher.get("publisher_identifier_scheme"),
        publisher.get("scheme_uri"),
        publisher.get("lang"),
    )
    if existing_id is not None:
        return int(existing_id)
    return int(await conn.fetchval(
        """
        INSERT INTO publishers (
            name, publisher_identifier, publisher_identifier_scheme, scheme_uri, lang
        )
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
        """,
        publisher["name"],
        publisher.get("publisher_identifier"),
        publisher.get("publisher_identifier_scheme"),
        publisher.get("scheme_uri"),
        publisher.get("lang"),
    ))


async def _find_duplicate_datasheet(conn: asyncpg.Connection, asset: AssetDatasheetCreate) -> int | None:
    primary_title = asset.titles[0].title if asset.titles else None
    primary_creator = asset.creators[0].creator_name if asset.creators else None
    row = await conn.fetchrow(
        """
        SELECT d.identifier
        FROM datasheets d
        LEFT JOIN LATERAL (
            SELECT title
            FROM datasheet_titles
            WHERE datasheet_id = d.identifier
            ORDER BY id
            LIMIT 1
        ) t ON TRUE
        LEFT JOIN LATERAL (
            SELECT creator_name
            FROM datasheet_creators
            WHERE datasheet_id = d.identifier
            ORDER BY id
            LIMIT 1
        ) c ON TRUE
        WHERE COALESCE(t.title, '') = COALESCE($1, '')
          AND COALESCE(c.creator_name, '') = COALESCE($2, '')
          AND COALESCE(d.version, '') = COALESCE($3, '')
          AND COALESCE(d.publication_year, -1) = COALESCE($4, -1)
        LIMIT 1
        """,
        primary_title,
        primary_creator,
        asset.version,
        asset.publication_year,
    )
    return int(row["identifier"]) if row else None


async def _insert_many(
    conn: asyncpg.Connection,
    query: str,
    rows: Sequence[tuple],
) -> None:
    if rows:
        await conn.executemany(query, rows)


async def _create_datasheet_in_tx(
    conn: asyncpg.Connection,
    asset: AssetDatasheetCreate,
    organization: str,
    revision_context: AssetRevisionContext | None = None,
) -> AssetIngestResult:
    duplicate_id = await _find_duplicate_datasheet(conn, asset)
    if duplicate_id is not None:
        return AssetIngestResult(
            asset_type="datasheet",
            asset_id=duplicate_id,
            organization=organization,
            created=False,
            duplicate=True,
        )

    publisher_id = await _find_publisher_id(conn, asset.publisher.model_dump(exclude_none=True) if asset.publisher else None)
    dataset_schema_id = asset.dataset_schema_id
    if dataset_schema_id is None and asset.dataset_schema_blob is not None:
        dataset_schema_id = await conn.fetchval(
            """
            INSERT INTO dataset_schemas (blob, created_at, updated_at)
            VALUES ($1::jsonb, NOW(), NOW())
            RETURNING id
            """,
            json.dumps(asset.dataset_schema_blob),
        )
    datasheet_id = await conn.fetchval(
        """
        INSERT INTO datasheets (
            publication_year, resource_type, resource_type_general, size, format, version,
            is_private, status, created_at, updated_at, dataset_schema_id, publisher_id,
            asset_version, previous_version_id, root_version_id
        )
        VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, 'approved', NOW(), NOW(), $8, $9,
            $10, $11, $12
        )
        RETURNING identifier
        """,
        asset.publication_year,
        asset.resource_type,
        asset.resource_type_general,
        asset.size,
        asset.format,
        asset.version,
        asset.is_private,
        dataset_schema_id,
        publisher_id,
        revision_context.asset_version if revision_context else 1,
        revision_context.previous_version_id if revision_context else None,
        revision_context.root_version_id if revision_context else None,
    )

    if revision_context is None:
        await conn.execute(
            """
            UPDATE datasheets
            SET root_version_id = $2
            WHERE identifier = $1
            """,
            datasheet_id,
            datasheet_id,
        )

    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_creators (
            datasheet_id, creator_name, name_type, lang, given_name, family_name,
            name_identifier, name_identifier_scheme, name_id_scheme_uri,
            affiliation, affiliation_identifier, affiliation_identifier_scheme, affiliation_scheme_uri
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """,
        [
            (
                datasheet_id,
                creator.creator_name,
                creator.name_type,
                creator.lang,
                creator.given_name,
                creator.family_name,
                creator.name_identifier,
                creator.name_identifier_scheme,
                creator.name_id_scheme_uri,
                creator.affiliation,
                creator.affiliation_identifier,
                creator.affiliation_identifier_scheme,
                creator.affiliation_scheme_uri,
            )
            for creator in asset.creators
        ],
    )
    await _insert_many(
        conn,
        "INSERT INTO datasheet_titles (datasheet_id, title, title_type, lang) VALUES ($1, $2, $3, $4)",
        [(datasheet_id, title.title, title.title_type, title.lang) for title in asset.titles],
    )
    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_subjects (
            datasheet_id, subject, subject_scheme, scheme_uri, value_uri, classification_code, lang
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        [
            (
                datasheet_id,
                subject.subject,
                subject.subject_scheme,
                subject.scheme_uri,
                subject.value_uri,
                subject.classification_code,
                subject.lang,
            )
            for subject in asset.subjects
        ],
    )
    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_contributors (
            datasheet_id, contributor_type, contributor_name, name_type, given_name, family_name,
            name_identifier, name_identifier_scheme, name_id_scheme_uri,
            affiliation, affiliation_identifier, affiliation_identifier_scheme, affiliation_scheme_uri
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """,
        [
            (
                datasheet_id,
                contributor.contributor_type,
                contributor.contributor_name,
                contributor.name_type,
                contributor.given_name,
                contributor.family_name,
                contributor.name_identifier,
                contributor.name_identifier_scheme,
                contributor.name_id_scheme_uri,
                contributor.affiliation,
                contributor.affiliation_identifier,
                contributor.affiliation_identifier_scheme,
                contributor.affiliation_scheme_uri,
            )
            for contributor in asset.contributors
        ],
    )
    await _insert_many(
        conn,
        "INSERT INTO datasheet_dates (datasheet_id, date, date_type, date_information) VALUES ($1, $2, $3, $4)",
        [(datasheet_id, item.date, item.date_type, item.date_information) for item in asset.dates],
    )
    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_alternate_identifiers (
            datasheet_id, alternate_identifier, alternate_identifier_type
        )
        VALUES ($1, $2, $3)
        """,
        [
            (datasheet_id, item.alternate_identifier, item.alternate_identifier_type)
            for item in asset.alternate_identifiers
        ],
    )
    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_related_identifiers (
            datasheet_id, related_identifier, related_identifier_type, relation_type,
            related_metadata_scheme, scheme_uri, scheme_type, resource_type_general
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        [
            (
                datasheet_id,
                item.related_identifier,
                item.related_identifier_type,
                item.relation_type,
                item.related_metadata_scheme,
                item.scheme_uri,
                item.scheme_type,
                item.resource_type_general,
            )
            for item in asset.related_identifiers
        ],
    )
    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_rights (
            datasheet_id, rights, rights_uri, rights_identifier, rights_identifier_scheme, scheme_uri, lang
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        [
            (
                datasheet_id,
                item.rights,
                item.rights_uri,
                item.rights_identifier,
                item.rights_identifier_scheme,
                item.scheme_uri,
                item.lang,
            )
            for item in asset.rights_list
        ],
    )
    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_descriptions (datasheet_id, description, description_type, lang)
        VALUES ($1, $2, $3, $4)
        """,
        [
            (datasheet_id, item.description, item.description_type, item.lang)
            for item in asset.descriptions
        ],
    )
    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_geo_locations (
            datasheet_id, geo_location_place, point_longitude, point_latitude,
            box_west, box_east, box_south, box_north, polygon
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
        """,
        [
            (
                datasheet_id,
                item.geo_location_place,
                item.point_longitude,
                item.point_latitude,
                item.box_west,
                item.box_east,
                item.box_south,
                item.box_north,
                json.dumps(item.polygon) if item.polygon is not None else None,
            )
            for item in asset.geo_locations
        ],
    )
    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_funding_references (
            datasheet_id, funder_name, funder_identifier, funder_identifier_type,
            scheme_uri, award_number, award_uri, award_title
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        [
            (
                datasheet_id,
                item.funder_name,
                item.funder_identifier,
                item.funder_identifier_type,
                item.scheme_uri,
                item.award_number,
                item.award_uri,
                item.award_title,
            )
            for item in asset.funding_references
        ],
    )

    log.info("Asset ingest: created datasheet %s for org %s", datasheet_id, organization)
    return AssetIngestResult(
        asset_type="datasheet",
        asset_id=int(datasheet_id),
        organization=organization,
        created=True,
    )


@router.post("/model-cards", response_model=AssetIngestResult, status_code=status.HTTP_201_CREATED)
async def create_model_card_asset(
    asset: AssetModelCardCreate,
    principal: AssetIngestPrincipal = Depends(require_asset_ingest_principal),
    pool: asyncpg.Pool = Depends(get_pool),
):
    async with pool.acquire() as conn:
        async with conn.transaction():
            result = await _create_model_card_in_tx(
                conn,
                asset,
                principal.organization,
                revision_context=_model_card_revision_context(asset),
            )
    if result.duplicate:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model card already exists with id {result.asset_id}",
        )
    return result


@router.post("/datasheets", response_model=AssetIngestResult, status_code=status.HTTP_201_CREATED)
async def create_datasheet_asset(
    asset: AssetDatasheetCreate,
    principal: AssetIngestPrincipal = Depends(require_asset_ingest_principal),
    pool: asyncpg.Pool = Depends(get_pool),
):
    async with pool.acquire() as conn:
        async with conn.transaction():
            result = await _create_datasheet_in_tx(
                conn,
                asset,
                principal.organization,
                revision_context=_datasheet_revision_context(asset),
            )
    if result.duplicate:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Datasheet already exists with id {result.asset_id}",
        )
    return result


@router.post("/model-cards/bulk", response_model=AssetBulkIngestResult)
async def bulk_create_model_card_assets(
    payload: AssetBulkModelCardCreate,
    principal: AssetIngestPrincipal = Depends(require_asset_ingest_principal),
    pool: asyncpg.Pool = Depends(get_pool),
):
    results: list[AssetBulkItemResult] = []
    async with pool.acquire() as conn:
        for index, asset in enumerate(payload.assets):
            try:
                async with conn.transaction():
                    result = await _create_model_card_in_tx(conn, asset, principal.organization)
                results.append(
                    AssetBulkItemResult(
                        index=index,
                        asset_type="model_card",
                        created=result.created,
                        duplicate=result.duplicate,
                        asset_id=result.asset_id,
                    )
                )
            except Exception as exc:
                log.exception("Bulk model card ingest failed at index %s", index)
                results.append(
                    AssetBulkItemResult(
                        index=index,
                        asset_type="model_card",
                        created=False,
                        error=str(exc),
                    )
                )
    return AssetBulkIngestResult(
        asset_type="model_card",
        organization=principal.organization,
        total=len(results),
        created=sum(1 for item in results if item.created),
        duplicates=sum(1 for item in results if item.duplicate),
        failed=sum(1 for item in results if item.error is not None),
        results=results,
    )


@router.post("/datasheets/bulk", response_model=AssetBulkIngestResult)
async def bulk_create_datasheet_assets(
    payload: AssetBulkDatasheetCreate,
    principal: AssetIngestPrincipal = Depends(require_asset_ingest_principal),
    pool: asyncpg.Pool = Depends(get_pool),
):
    results: list[AssetBulkItemResult] = []
    async with pool.acquire() as conn:
        for index, asset in enumerate(payload.assets):
            try:
                async with conn.transaction():
                    result = await _create_datasheet_in_tx(conn, asset, principal.organization)
                results.append(
                    AssetBulkItemResult(
                        index=index,
                        asset_type="datasheet",
                        created=result.created,
                        duplicate=result.duplicate,
                        asset_id=result.asset_id,
                    )
                )
            except Exception as exc:
                log.exception("Bulk datasheet ingest failed at index %s", index)
                results.append(
                    AssetBulkItemResult(
                        index=index,
                        asset_type="datasheet",
                        created=False,
                        error=str(exc),
                    )
                )
    return AssetBulkIngestResult(
        asset_type="datasheet",
        organization=principal.organization,
        total=len(results),
        created=sum(1 for item in results if item.created),
        duplicates=sum(1 for item in results if item.duplicate),
        failed=sum(1 for item in results if item.error is not None),
        results=results,
    )


async def _fetch_model_card_snapshot(conn: asyncpg.Connection, asset_id: int) -> dict | None:
    card = await conn.fetchrow(
        """
        SELECT id, name, version, is_private, is_gated, status,
               short_description, full_description, keywords, author, citation,
               input_data, input_type, output_data, foundational_model, category, documentation,
               created_at, updated_at, asset_version, previous_version_id,
               COALESCE(root_version_id, id) AS root_version_id
        FROM model_cards
        WHERE id = $1
        """,
        asset_id,
    )
    if not card:
        return None

    model = await conn.fetchrow(
        """
        SELECT id, name, version, description, owner, location, license, framework,
               model_type, test_accuracy, model_metrics, inference_labels, model_structure,
               created_at, updated_at
        FROM models
        WHERE model_card_id = $1
        LIMIT 1
        """,
        asset_id,
    )
    return {
        "asset_type": "model_card",
        "asset_id": int(card["id"]),
        "asset_version": int(card["asset_version"] or 1),
        "card": dict(card),
        "model": dict(model) if model else None,
    }


async def _fetch_datasheet_snapshot(conn: asyncpg.Connection, asset_id: int) -> dict | None:
    datasheet = await conn.fetchrow(
        """
        SELECT d.identifier, d.publication_year, d.resource_type, d.resource_type_general,
               d.size, d.format, d.version, d.is_private, d.status, d.created_at, d.updated_at,
               d.dataset_schema_id, d.publisher_id, d.asset_version, d.previous_version_id,
               COALESCE(d.root_version_id, d.identifier) AS root_version_id,
               p.name AS publisher_name, p.publisher_identifier, p.publisher_identifier_scheme,
               p.scheme_uri AS publisher_scheme_uri, p.lang AS publisher_lang,
               ds.blob AS dataset_schema_blob
        FROM datasheets d
        LEFT JOIN publishers p ON p.id = d.publisher_id
        LEFT JOIN dataset_schemas ds ON ds.id = d.dataset_schema_id
        WHERE d.identifier = $1
        """,
        asset_id,
    )
    if not datasheet:
        return None

    tables = {
        "creators": "SELECT * FROM datasheet_creators WHERE datasheet_id = $1 ORDER BY id",
        "titles": "SELECT * FROM datasheet_titles WHERE datasheet_id = $1 ORDER BY id",
        "subjects": "SELECT * FROM datasheet_subjects WHERE datasheet_id = $1 ORDER BY id",
        "contributors": "SELECT * FROM datasheet_contributors WHERE datasheet_id = $1 ORDER BY id",
        "dates": "SELECT * FROM datasheet_dates WHERE datasheet_id = $1 ORDER BY id",
        "alternate_identifiers": "SELECT * FROM datasheet_alternate_identifiers WHERE datasheet_id = $1 ORDER BY id",
        "related_identifiers": "SELECT * FROM datasheet_related_identifiers WHERE datasheet_id = $1 ORDER BY id",
        "rights_list": "SELECT * FROM datasheet_rights WHERE datasheet_id = $1 ORDER BY id",
        "descriptions": "SELECT * FROM datasheet_descriptions WHERE datasheet_id = $1 ORDER BY id",
        "geo_locations": "SELECT * FROM datasheet_geo_locations WHERE datasheet_id = $1 ORDER BY id",
        "funding_references": "SELECT * FROM datasheet_funding_references WHERE datasheet_id = $1 ORDER BY id",
    }
    nested: dict[str, list[dict]] = {}
    for key, query in tables.items():
        rows = await conn.fetch(query, asset_id)
        nested[key] = [dict(row) for row in rows]

    return {
        "asset_type": "datasheet",
        "asset_id": int(datasheet["identifier"]),
        "asset_version": int(datasheet["asset_version"] or 1),
        "core": dict(datasheet),
        **nested,
    }


async def _fetch_asset_snapshot(conn: asyncpg.Connection, asset_type: str, asset_id: int) -> dict | None:
    if asset_type == "model_card":
        return await _fetch_model_card_snapshot(conn, asset_id)
    return await _fetch_datasheet_snapshot(conn, asset_id)


async def _update_model_card_in_tx(
    conn: asyncpg.Connection,
    asset_id: int,
    asset: AssetModelCardCreate,
    organization: str,
    changed_by: str | None,
) -> AssetUpdateResult:
    existing = await _fetch_model_card_snapshot(conn, asset_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Model card not found")

    await ensure_initial_backup(conn, "model_card", asset_id, int(existing["asset_version"]), existing)
    field_changes = _build_field_changes(
        _model_card_edit_state_from_snapshot(existing),
        _model_card_edit_state_from_asset(asset),
    )

    next_version = int(existing["asset_version"]) + 1
    await conn.execute(
        """
        UPDATE model_cards
        SET name = $2,
            version = $3,
            is_private = $4,
            is_gated = $5,
            short_description = $6,
            full_description = $7,
            keywords = $8,
            author = $9,
            citation = $10,
            input_data = $11,
            input_type = $12,
            output_data = $13,
            foundational_model = $14,
            category = $15,
            documentation = $16,
            asset_version = $17,
            updated_at = NOW()
        WHERE id = $1
        """,
        asset_id,
        asset.name,
        asset.version,
        asset.is_private,
        asset.is_gated,
        asset.short_description,
        asset.full_description,
        asset.keywords,
        asset.author,
        asset.citation,
        asset.input_data,
        asset.input_type,
        asset.output_data,
        asset.foundational_model,
        asset.category,
        asset.documentation,
        next_version,
    )

    model_row_id = await conn.fetchval("SELECT id FROM models WHERE model_card_id = $1 LIMIT 1", asset_id)
    if asset.ai_model is not None:
        if model_row_id is None:
            await conn.execute(
                """
                INSERT INTO models (
                    name, version, description, owner, location, license, framework, model_type,
                    test_accuracy, model_metrics, inference_labels, model_structure,
                    created_at, updated_at, model_card_id
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8,
                    $9, $10::jsonb, $11::jsonb, $12::jsonb,
                    NOW(), NOW(), $13
                )
                """,
                asset.ai_model.name,
                asset.ai_model.version,
                asset.ai_model.description,
                asset.ai_model.owner,
                asset.ai_model.location,
                asset.ai_model.license,
                asset.ai_model.framework,
                asset.ai_model.model_type,
                asset.ai_model.test_accuracy,
                json.dumps(asset.ai_model.model_metrics),
                json.dumps(asset.ai_model.inference_labels),
                json.dumps(asset.ai_model.model_structure),
                asset_id,
            )
        else:
            await conn.execute(
                """
                UPDATE models
                SET name = $2,
                    version = $3,
                    description = $4,
                    owner = $5,
                    location = $6,
                    license = $7,
                    framework = $8,
                    model_type = $9,
                    test_accuracy = $10,
                    model_metrics = $11::jsonb,
                    inference_labels = $12::jsonb,
                    model_structure = $13::jsonb,
                    updated_at = NOW()
                WHERE id = $1
                """,
                model_row_id,
                asset.ai_model.name,
                asset.ai_model.version,
                asset.ai_model.description,
                asset.ai_model.owner,
                asset.ai_model.location,
                asset.ai_model.license,
                asset.ai_model.framework,
                asset.ai_model.model_type,
                asset.ai_model.test_accuracy,
                json.dumps(asset.ai_model.model_metrics),
                json.dumps(asset.ai_model.inference_labels),
                json.dumps(asset.ai_model.model_structure),
            )

    updated_snapshot = await _fetch_model_card_snapshot(conn, asset_id)
    backup_id = await record_backup(conn, "model_card", asset_id, next_version, "update", updated_snapshot)
    await _record_change_log(conn, "model_card", asset_id, next_version, changed_by, field_changes)
    return AssetUpdateResult(
        asset_type="model_card",
        asset_id=asset_id,
        organization=organization,
        asset_version=next_version,
        backup_id=backup_id,
    )


async def _replace_datasheet_children(conn: asyncpg.Connection, datasheet_id: int, asset: AssetDatasheetCreate) -> None:
    for table_name in (
        "datasheet_creators",
        "datasheet_titles",
        "datasheet_subjects",
        "datasheet_contributors",
        "datasheet_dates",
        "datasheet_alternate_identifiers",
        "datasheet_related_identifiers",
        "datasheet_rights",
        "datasheet_descriptions",
        "datasheet_geo_locations",
        "datasheet_funding_references",
    ):
        await conn.execute(f"DELETE FROM {table_name} WHERE datasheet_id = $1", datasheet_id)

    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_creators (
            datasheet_id, creator_name, name_type, lang, given_name, family_name,
            name_identifier, name_identifier_scheme, name_id_scheme_uri,
            affiliation, affiliation_identifier, affiliation_identifier_scheme, affiliation_scheme_uri
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """,
        [
            (
                datasheet_id,
                creator.creator_name,
                creator.name_type,
                creator.lang,
                creator.given_name,
                creator.family_name,
                creator.name_identifier,
                creator.name_identifier_scheme,
                creator.name_id_scheme_uri,
                creator.affiliation,
                creator.affiliation_identifier,
                creator.affiliation_identifier_scheme,
                creator.affiliation_scheme_uri,
            )
            for creator in asset.creators
        ],
    )
    await _insert_many(conn, "INSERT INTO datasheet_titles (datasheet_id, title, title_type, lang) VALUES ($1, $2, $3, $4)",
        [(datasheet_id, title.title, title.title_type, title.lang) for title in asset.titles])
    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_subjects (
            datasheet_id, subject, subject_scheme, scheme_uri, value_uri, classification_code, lang
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        [(datasheet_id, subject.subject, subject.subject_scheme, subject.scheme_uri, subject.value_uri, subject.classification_code, subject.lang) for subject in asset.subjects],
    )
    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_contributors (
            datasheet_id, contributor_type, contributor_name, name_type, given_name, family_name,
            name_identifier, name_identifier_scheme, name_id_scheme_uri,
            affiliation, affiliation_identifier, affiliation_identifier_scheme, affiliation_scheme_uri
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """,
        [(datasheet_id, c.contributor_type, c.contributor_name, c.name_type, c.given_name, c.family_name, c.name_identifier, c.name_identifier_scheme, c.name_id_scheme_uri, c.affiliation, c.affiliation_identifier, c.affiliation_identifier_scheme, c.affiliation_scheme_uri) for c in asset.contributors],
    )
    await _insert_many(conn, "INSERT INTO datasheet_dates (datasheet_id, date, date_type, date_information) VALUES ($1, $2, $3, $4)",
        [(datasheet_id, item.date, item.date_type, item.date_information) for item in asset.dates])
    await _insert_many(conn, "INSERT INTO datasheet_alternate_identifiers (datasheet_id, alternate_identifier, alternate_identifier_type) VALUES ($1, $2, $3)",
        [(datasheet_id, item.alternate_identifier, item.alternate_identifier_type) for item in asset.alternate_identifiers])
    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_related_identifiers (
            datasheet_id, related_identifier, related_identifier_type, relation_type,
            related_metadata_scheme, scheme_uri, scheme_type, resource_type_general
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        [(datasheet_id, item.related_identifier, item.related_identifier_type, item.relation_type, item.related_metadata_scheme, item.scheme_uri, item.scheme_type, item.resource_type_general) for item in asset.related_identifiers],
    )
    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_rights (
            datasheet_id, rights, rights_uri, rights_identifier, rights_identifier_scheme, scheme_uri, lang
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        [(datasheet_id, item.rights, item.rights_uri, item.rights_identifier, item.rights_identifier_scheme, item.scheme_uri, item.lang) for item in asset.rights_list],
    )
    await _insert_many(conn, "INSERT INTO datasheet_descriptions (datasheet_id, description, description_type, lang) VALUES ($1, $2, $3, $4)",
        [(datasheet_id, item.description, item.description_type, item.lang) for item in asset.descriptions])
    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_geo_locations (
            datasheet_id, geo_location_place, point_longitude, point_latitude,
            box_west, box_east, box_south, box_north, polygon
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
        """,
        [(datasheet_id, item.geo_location_place, item.point_longitude, item.point_latitude, item.box_west, item.box_east, item.box_south, item.box_north, json.dumps(item.polygon) if item.polygon is not None else None) for item in asset.geo_locations],
    )
    await _insert_many(
        conn,
        """
        INSERT INTO datasheet_funding_references (
            datasheet_id, funder_name, funder_identifier, funder_identifier_type,
            scheme_uri, award_number, award_uri, award_title
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        [(datasheet_id, item.funder_name, item.funder_identifier, item.funder_identifier_type, item.scheme_uri, item.award_number, item.award_uri, item.award_title) for item in asset.funding_references],
    )


async def _update_datasheet_in_tx(
    conn: asyncpg.Connection,
    asset_id: int,
    asset: AssetDatasheetCreate,
    organization: str,
    changed_by: str | None,
) -> AssetUpdateResult:
    existing = await _fetch_datasheet_snapshot(conn, asset_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Datasheet not found")

    await ensure_initial_backup(conn, "datasheet", asset_id, int(existing["asset_version"]), existing)
    field_changes = _build_field_changes(
        _datasheet_edit_state_from_snapshot(existing),
        _datasheet_edit_state_from_asset(asset),
    )

    publisher_id = await _find_publisher_id(conn, asset.publisher.model_dump(exclude_none=True) if asset.publisher else None)
    dataset_schema_id = asset.dataset_schema_id
    if dataset_schema_id is None and asset.dataset_schema_blob is not None:
        dataset_schema_id = await conn.fetchval(
            """
            INSERT INTO dataset_schemas (blob, created_at, updated_at)
            VALUES ($1::jsonb, NOW(), NOW())
            RETURNING id
            """,
            json.dumps(asset.dataset_schema_blob),
        )

    next_version = int(existing["asset_version"]) + 1
    await conn.execute(
        """
        UPDATE datasheets
        SET publication_year = $2,
            resource_type = $3,
            resource_type_general = $4,
            size = $5,
            format = $6,
            version = $7,
            is_private = $8,
            dataset_schema_id = $9,
            publisher_id = $10,
            asset_version = $11,
            updated_at = NOW()
        WHERE identifier = $1
        """,
        asset_id,
        asset.publication_year,
        asset.resource_type,
        asset.resource_type_general,
        asset.size,
        asset.format,
        asset.version,
        asset.is_private,
        dataset_schema_id,
        publisher_id,
        next_version,
    )
    await _replace_datasheet_children(conn, asset_id, asset)
    updated_snapshot = await _fetch_datasheet_snapshot(conn, asset_id)
    backup_id = await record_backup(conn, "datasheet", asset_id, next_version, "update", updated_snapshot)
    await _record_change_log(conn, "datasheet", asset_id, next_version, changed_by, field_changes)
    return AssetUpdateResult(
        asset_type="datasheet",
        asset_id=asset_id,
        organization=organization,
        asset_version=next_version,
        backup_id=backup_id,
    )


@router.get("/records", response_model=list[EditableRecordSummary])
async def list_editable_records(
    principal: AssetIngestPrincipal = Depends(require_asset_ingest_principal),
    pool: asyncpg.Pool = Depends(get_pool),
    q: str | None = Query(default=None, max_length=255),
    limit: int = Query(20, ge=1, le=100),
):
    query_text = q.strip() if q else None
    params: list[object] = []
    model_filter = ""
    datasheet_filter = ""
    if query_text:
        params.append(f"%{query_text}%")
        model_filter = f"""
            AND (
                mc.name ILIKE ${len(params)}
                OR COALESCE(mc.author, '') ILIKE ${len(params)}
                OR COALESCE(mc.short_description, '') ILIKE ${len(params)}
                OR COALESCE(mc.full_description, '') ILIKE ${len(params)}
                OR COALESCE(mc.keywords, '') ILIKE ${len(params)}
                OR COALESCE(mc.category, '') ILIKE ${len(params)}
                OR COALESCE(mc.input_data, '') ILIKE ${len(params)}
                OR COALESCE(mc.output_data, '') ILIKE ${len(params)}
                OR COALESCE(mc.citation, '') ILIKE ${len(params)}
                OR COALESCE(mc.foundational_model, '') ILIKE ${len(params)}
                OR COALESCE(mc.documentation, '') ILIKE ${len(params)}
                OR COALESCE(m.framework, '') ILIKE ${len(params)}
                OR COALESCE(m.license, '') ILIKE ${len(params)}
                OR COALESCE(m.owner, '') ILIKE ${len(params)}
                OR COALESCE(m.model_type, '') ILIKE ${len(params)}
            )
        """
        datasheet_filter = f"""
            AND (
                EXISTS (SELECT 1 FROM datasheet_titles dt WHERE dt.datasheet_id = d.identifier AND dt.title ILIKE ${len(params)})
                OR EXISTS (SELECT 1 FROM datasheet_creators dc WHERE dc.datasheet_id = d.identifier AND dc.creator_name ILIKE ${len(params)})
                OR EXISTS (SELECT 1 FROM datasheet_descriptions dd WHERE dd.datasheet_id = d.identifier AND dd.description ILIKE ${len(params)})
                OR EXISTS (SELECT 1 FROM datasheet_subjects ds WHERE ds.datasheet_id = d.identifier AND ds.subject ILIKE ${len(params)})
                OR EXISTS (SELECT 1 FROM datasheet_contributors dco WHERE dco.datasheet_id = d.identifier AND dco.contributor_name ILIKE ${len(params)})
                OR EXISTS (SELECT 1 FROM datasheet_related_identifiers dri WHERE dri.datasheet_id = d.identifier AND dri.related_identifier ILIKE ${len(params)})
                OR EXISTS (SELECT 1 FROM datasheet_alternate_identifiers dai WHERE dai.datasheet_id = d.identifier AND dai.alternate_identifier ILIKE ${len(params)})
                OR COALESCE(p.name, '') ILIKE ${len(params)}
                OR COALESCE(d.resource_type, '') ILIKE ${len(params)}
                OR COALESCE(d.resource_type_general, '') ILIKE ${len(params)}
                OR COALESCE(d.size, '') ILIKE ${len(params)}
                OR COALESCE(d.format, '') ILIKE ${len(params)}
                OR COALESCE(d.version, '') ILIKE ${len(params)}
            )
        """
    params.append(limit)
    query = f"""
        WITH latest_model_cards AS (
            SELECT *
            FROM (
                SELECT mc.id, mc.name, mc.author, mc.short_description, mc.full_description, mc.keywords, mc.category, mc.updated_at,
                       ROW_NUMBER() OVER (PARTITION BY COALESCE(mc.root_version_id, mc.id) ORDER BY mc.asset_version DESC, mc.id DESC) AS rn
                FROM model_cards mc
                LEFT JOIN models m ON m.model_card_id = mc.id
                WHERE mc.status = 'approved'
                {model_filter}
            ) ranked
            WHERE rn = 1
        ),
        latest_datasheets AS (
            SELECT *
            FROM (
                SELECT d.identifier, d.updated_at, d.resource_type, d.version, p.name AS publisher_name,
                       ROW_NUMBER() OVER (PARTITION BY COALESCE(d.root_version_id, d.identifier) ORDER BY d.asset_version DESC, d.identifier DESC) AS rn
                FROM datasheets d
                LEFT JOIN publishers p ON p.id = d.publisher_id
                WHERE d.status = 'approved'
                {datasheet_filter}
            ) ranked
            WHERE rn = 1
        ),
        datasheet_titles_first AS (
            SELECT DISTINCT ON (datasheet_id) datasheet_id, title
            FROM datasheet_titles
            ORDER BY datasheet_id, id
        ),
        datasheet_creators_first AS (
            SELECT DISTINCT ON (datasheet_id) datasheet_id, creator_name
            FROM datasheet_creators
            ORDER BY datasheet_id, id
        ),
        datasheet_descriptions_first AS (
            SELECT DISTINCT ON (datasheet_id) datasheet_id, description
            FROM datasheet_descriptions
            ORDER BY datasheet_id, id
        )
        SELECT *
        FROM (
            SELECT
                'model_card' AS asset_type,
                lmc.id AS asset_id,
                lmc.name AS title,
                lmc.author AS subtitle,
                COALESCE(lmc.short_description, lmc.full_description, lmc.category) AS description,
                'Model Card' AS kind_label,
                lmc.updated_at
            FROM latest_model_cards lmc
            UNION ALL
            SELECT
                'datasheet' AS asset_type,
                ld.identifier AS asset_id,
                COALESCE(dtf.title, 'Untitled datasheet') AS title,
                COALESCE(dcf.creator_name, ld.publisher_name, 'Published datasheet') AS subtitle,
                COALESCE(ddf.description, ld.resource_type) AS description,
                'Datasheet' AS kind_label,
                ld.updated_at
            FROM latest_datasheets ld
            LEFT JOIN datasheet_titles_first dtf ON dtf.datasheet_id = ld.identifier
            LEFT JOIN datasheet_creators_first dcf ON dcf.datasheet_id = ld.identifier
            LEFT JOIN datasheet_descriptions_first ddf ON ddf.datasheet_id = ld.identifier
        ) combined
        ORDER BY updated_at DESC NULLS LAST, LOWER(title)
        LIMIT ${len(params)}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    return [
        EditableRecordSummary(
            asset_type=row["asset_type"],
            asset_id=int(row["asset_id"]),
            title=row["title"],
            subtitle=row["subtitle"],
            description=row["description"],
            kind_label=row["kind_label"],
            updated_at=row["updated_at"].isoformat() if row["updated_at"] else None,
        )
        for row in rows
    ]


@router.patch("/model-cards/{asset_id}", response_model=AssetUpdateResult)
async def update_model_card_asset(
    request: Request,
    asset_id: int = Path(..., ge=1),
    asset: AssetModelCardCreate = ...,
    principal: AssetIngestPrincipal = Depends(require_asset_ingest_principal),
    pool: asyncpg.Pool = Depends(get_pool),
):
    actor = get_request_actor(request)
    async with pool.acquire() as conn:
        async with conn.transaction():
            return await _update_model_card_in_tx(conn, asset_id, asset, principal.organization, actor.username)


@router.patch("/datasheets/{asset_id}", response_model=AssetUpdateResult)
async def update_datasheet_asset(
    request: Request,
    asset_id: int = Path(..., ge=1),
    asset: AssetDatasheetCreate = ...,
    principal: AssetIngestPrincipal = Depends(require_asset_ingest_principal),
    pool: asyncpg.Pool = Depends(get_pool),
):
    actor = get_request_actor(request)
    async with pool.acquire() as conn:
        async with conn.transaction():
            return await _update_datasheet_in_tx(conn, asset_id, asset, principal.organization, actor.username)


@router.get("/backups/{asset_type}/{asset_id}", response_model=list[AssetBackupRecord])
async def list_asset_backups(
    asset_type: str = Path(..., pattern="^(model_card|datasheet)$"),
    asset_id: int = Path(..., ge=1),
    principal: AssetIngestPrincipal = Depends(require_asset_ingest_principal),
    pool: asyncpg.Pool = Depends(get_pool),
):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, asset_type, asset_id, asset_version, backup_kind, sequence, file_path, created_at
            FROM asset_backups
            WHERE asset_type = $1 AND asset_id = $2
            ORDER BY sequence DESC
            """,
            asset_type,
            asset_id,
        )
    return [
        AssetBackupRecord(
            id=int(row["id"]),
            asset_type=row["asset_type"],
            asset_id=int(row["asset_id"]),
            asset_version=int(row["asset_version"]),
            backup_kind=row["backup_kind"],
            sequence=int(row["sequence"]),
            file_path=row["file_path"],
            created_at=row["created_at"].isoformat(),
        )
        for row in rows
    ]


@router.get("/changelog/{asset_type}/{asset_id}", response_model=list[AssetChangeLogEntry])
async def list_asset_changelog(
    asset_type: str = Path(..., pattern="^(model_card|datasheet)$"),
    asset_id: int = Path(..., ge=1),
    principal: AssetIngestPrincipal = Depends(require_asset_ingest_principal),
    pool: asyncpg.Pool = Depends(get_pool),
    limit: int = Query(25, ge=1, le=100),
):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, asset_type, asset_id, asset_version, changed_by, changed_at, summary, changes
            FROM asset_change_logs
            WHERE asset_type = $1 AND asset_id = $2
            ORDER BY changed_at DESC, id DESC
            LIMIT $3
            """,
            asset_type,
            asset_id,
            limit,
        )
    return [
        AssetChangeLogEntry(
            id=int(row["id"]),
            asset_type=row["asset_type"],
            asset_id=int(row["asset_id"]),
            asset_version=int(row["asset_version"]),
            changed_by=row["changed_by"],
            changed_at=row["changed_at"].isoformat(),
            summary=row["summary"],
            changes=[AssetFieldChange(**item) for item in _coerce_change_items(row["changes"])],
        )
        for row in rows
    ]


async def _latest_model_card_ids(conn: asyncpg.Connection) -> list[int]:
    rows = await conn.fetch(
        """
        WITH ranked AS (
            SELECT id,
                   ROW_NUMBER() OVER (PARTITION BY COALESCE(root_version_id, id) ORDER BY asset_version DESC, id DESC) AS rn
            FROM model_cards
            WHERE status = 'approved'
        )
        SELECT id
        FROM ranked
        WHERE rn = 1
        """
    )
    return [int(row["id"]) for row in rows]


async def _latest_datasheet_ids(conn: asyncpg.Connection) -> list[int]:
    rows = await conn.fetch(
        """
        WITH ranked AS (
            SELECT identifier,
                   ROW_NUMBER() OVER (PARTITION BY COALESCE(root_version_id, identifier) ORDER BY asset_version DESC, identifier DESC) AS rn
            FROM datasheets
            WHERE status = 'approved'
        )
        SELECT identifier
        FROM ranked
        WHERE rn = 1
        """
    )
    return [int(row["identifier"]) for row in rows]


async def run_periodic_backup_once(pool: asyncpg.Pool) -> AssetBackupRunResult:
    total_assets = 0
    created_backups = 0
    async with pool.acquire() as conn:
        async with conn.transaction():
            for asset_id in await _latest_model_card_ids(conn):
                snapshot = await _fetch_model_card_snapshot(conn, asset_id)
                if snapshot is not None:
                    total_assets += 1
                    await record_backup(conn, "model_card", asset_id, int(snapshot["asset_version"]), "periodic", snapshot)
                    created_backups += 1
            for asset_id in await _latest_datasheet_ids(conn):
                snapshot = await _fetch_datasheet_snapshot(conn, asset_id)
                if snapshot is not None:
                    total_assets += 1
                    await record_backup(conn, "datasheet", asset_id, int(snapshot["asset_version"]), "periodic", snapshot)
                    created_backups += 1
    return AssetBackupRunResult(
        backup_kind="periodic",
        total_assets=total_assets,
        created_backups=created_backups,
    )


@router.post("/backups/run", response_model=AssetBackupRunResult)
async def run_periodic_backup_endpoint(
    principal: AssetIngestPrincipal = Depends(require_asset_ingest_principal),
    pool: asyncpg.Pool = Depends(get_pool),
):
    return await run_periodic_backup_once(pool)
