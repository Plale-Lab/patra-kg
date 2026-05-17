import asyncio
import logging
from pathlib import Path

import asyncpg

from shared.config import get_database_url
from shared.db import build_connection_options, _MAX_RETRIES, _RETRY_DELAY_S
from rest_server.errors import database_unavailable

log = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None
_SCHEMA_FILE = Path(__file__).resolve().parents[1] / "db" / "bootstrap_schema.sql"


async def init_pool() -> asyncpg.Pool:
    """Create connection pool with retries. Called during app lifespan startup."""
    global _pool
    url = get_database_url()
    if not url:
        raise ValueError("DATABASE_URL environment variable is required")

    dsn, ssl_arg, direct_tls = build_connection_options(url)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            _pool = await asyncpg.create_pool(
                dsn,
                ssl=ssl_arg,
                direct_tls=direct_tls,
                min_size=1,
                max_size=10,
                command_timeout=60,
                timeout=30,
            )
            await ensure_schema(_pool)
            log.info("Database pool ready (attempt %d)", attempt)
            return _pool
        except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
            if attempt == _MAX_RETRIES:
                raise
            log.warning(
                "DB connection attempt %d/%d failed (%s), retrying in %ds …",
                attempt, _MAX_RETRIES, exc, _RETRY_DELAY_S,
            )
            await asyncio.sleep(_RETRY_DELAY_S)
    raise RuntimeError("Unreachable")


async def ensure_schema(pool: asyncpg.Pool) -> None:
    if not _SCHEMA_FILE.exists():
        raise RuntimeError(f"Bootstrap schema file not found: {_SCHEMA_FILE}")

    sql = _SCHEMA_FILE.read_text(encoding="utf-8")
    async with pool.acquire() as conn:
        await conn.execute(sql)
    log.info("Database schema bootstrap ensured")


async def close_pool() -> None:
    """Close connection pool. Called during app lifespan shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    """FastAPI dependency: returns the connection pool."""
    if _pool is None:
        raise database_unavailable()
    return _pool
