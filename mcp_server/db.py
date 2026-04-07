"""asyncpg connection pool for the MCP server (read-only, standalone)."""

import asyncio
import logging
import os
from datetime import date, datetime
from decimal import Decimal

import asyncpg

log = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None
_MAX_RETRIES = 5
_RETRY_DELAY_S = 3

DOMAIN_TABLES = {
    "animal-ecology": {
        "events": "camera_trap_events",
        "power": "camera_trap_power",
    },
    "digital-ag": {
        "events": "digital_ag_events",
        "power": "digital_ag_power",
    },
}


def _serialize_row(record: asyncpg.Record | None) -> dict | None:
    """Convert an asyncpg Record to a JSON-safe dict."""
    if record is None:
        return None
    d = dict(record)
    for k, v in d.items():
        if isinstance(v, Decimal):
            d[k] = float(v)
        elif isinstance(v, (datetime, date)):
            d[k] = v.isoformat()
    return d


async def init_pool() -> asyncpg.Pool:
    """Create connection pool with retries."""
    global _pool
    if _pool is not None:
        return _pool

    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable is required")

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            _pool = await asyncpg.create_pool(
                url,
                min_size=1,
                max_size=5,
                command_timeout=60,
                timeout=30,
            )
            log.info("MCP database pool ready (attempt %d)", attempt)
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


async def close_pool() -> None:
    """Close connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    """Return the connection pool; raise if not initialised."""
    if _pool is None:
        raise RuntimeError("Database pool not initialized")
    return _pool
