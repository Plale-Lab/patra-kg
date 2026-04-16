import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import asyncpg

from shared.config import get_asset_backup_dir


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def get_backup_dir() -> Path:
    return Path(get_asset_backup_dir()).expanduser()


async def ensure_initial_backup(conn: asyncpg.Connection, asset_type: str, asset_id: int, asset_version: int, snapshot: dict) -> int | None:
    existing = await conn.fetchval(
        """
        SELECT id
        FROM asset_backups
        WHERE asset_type = $1 AND asset_id = $2 AND backup_kind = 'initial'
        LIMIT 1
        """,
        asset_type,
        asset_id,
    )
    if existing is not None:
        return None
    return await record_backup(conn, asset_type, asset_id, asset_version, "initial", snapshot)


async def record_backup(
    conn: asyncpg.Connection,
    asset_type: str,
    asset_id: int,
    asset_version: int,
    backup_kind: str,
    snapshot: dict,
) -> int:
    next_sequence = await conn.fetchval(
        """
        SELECT COALESCE(MAX(sequence), 0) + 1
        FROM asset_backups
        WHERE asset_type = $1 AND asset_id = $2
        """,
        asset_type,
        asset_id,
    )
    file_path = _write_backup_file(asset_type, asset_id, int(next_sequence), backup_kind, snapshot)
    backup_id = await conn.fetchval(
        """
        INSERT INTO asset_backups (
            asset_type, asset_id, asset_version, backup_kind, sequence, snapshot, file_path, created_at
        )
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, NOW())
        RETURNING id
        """,
        asset_type,
        asset_id,
        asset_version,
        backup_kind,
        int(next_sequence),
        json.dumps(snapshot, default=_json_default),
        file_path,
    )
    return int(backup_id)


def _write_backup_file(asset_type: str, asset_id: int, sequence: int, backup_kind: str, snapshot: dict) -> str:
    backup_dir = get_backup_dir() / asset_type / str(asset_id)
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    file_path = backup_dir / f"{sequence:04d}-{backup_kind}-{timestamp}.json"
    file_path.write_text(json.dumps(snapshot, ensure_ascii=True, indent=2, default=_json_default), encoding="utf-8")
    return str(file_path)
