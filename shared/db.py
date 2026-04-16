"""Shared database connection helpers for REST and MCP servers."""

import ssl
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

_TAPIS_PODS_SUFFIX = ".pods.icicleai.tapis.io"
_TAPIS_PG_PORT = 443
_MAX_RETRIES = 5
_RETRY_DELAY_S = 3


def build_connection_options(raw_url: str) -> tuple[str, ssl.SSLContext | bool, bool]:
    """Normalise asyncpg connection options for the active PostgreSQL backend.

    Tapis Pods exposes PostgreSQL behind a 443 endpoint. If the host looks
    like a Tapis pod and the port is 5432, rewrite it to 443.

    asyncpg doesn't reliably consume the ``sslmode`` query-string parameter
    the way libpq/psycopg2 does, so we strip it and pass ``ssl`` explicitly.
    """
    parsed = urlparse(raw_url)

    host = parsed.hostname or ""
    port = parsed.port
    is_tapis_pod = host.endswith(_TAPIS_PODS_SUFFIX)
    if is_tapis_pod and port in (5432, None):
        netloc = parsed.netloc.replace(f":{port}", f":{_TAPIS_PG_PORT}", 1) if port else f"{parsed.netloc}:{_TAPIS_PG_PORT}"
        parsed = parsed._replace(netloc=netloc)

    qs = parse_qs(parsed.query)
    sslmode = qs.pop("sslmode", [None])[0]

    clean_url = urlunparse(parsed._replace(query=urlencode(qs, doseq=True)))

    if sslmode in ("require", "prefer"):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return clean_url, ctx, False
    if sslmode in ("verify-ca", "verify-full"):
        return clean_url, ssl.create_default_context(), False
    return clean_url, False, False
