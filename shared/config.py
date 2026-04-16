"""Centralised environment configuration for the Patra backend.

All functions read ``os.getenv`` at call time (not import time) so that
``monkeypatch.setenv`` in tests works transparently.
"""

import os


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_database_url() -> str | None:
    return os.getenv("DATABASE_URL")


# ---------------------------------------------------------------------------
# REST server startup
# ---------------------------------------------------------------------------

def get_db_startup_timeout_seconds() -> int:
    return int(os.getenv("DB_STARTUP_TIMEOUT_SECONDS", "12") or "12")


def get_asset_periodic_backup_interval_seconds() -> int:
    return int(os.getenv("ASSET_PERIODIC_BACKUP_INTERVAL_SECONDS", "0") or "0")


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() == "true"


def is_ask_patra_enabled() -> bool:
    return _env_flag("ENABLE_ASK_PATRA", default=False)


def is_automated_ingestion_enabled() -> bool:
    return _env_flag("ENABLE_AUTOMATED_INGESTION", default=False)


def is_domain_experiments_enabled() -> bool:
    return _env_flag("ENABLE_DOMAIN_EXPERIMENTS", default=False)


# ---------------------------------------------------------------------------
# Auth / admin
# ---------------------------------------------------------------------------

_DEFAULT_ADMIN_USERS = frozenset({"williamq96"})


def get_admin_users_csv() -> str:
    return os.getenv("PATRA_ADMIN_USERS", "").strip()


def get_default_admin_users() -> frozenset[str]:
    return _DEFAULT_ADMIN_USERS


def get_asset_ingest_keys_json() -> str:
    return os.getenv("PATRA_ASSET_INGEST_KEYS_JSON", "").strip()


# ---------------------------------------------------------------------------
# LLM / Agent defaults  (shared across agent-tools and ingestion)
# ---------------------------------------------------------------------------

def get_llm_api_base() -> str:
    return os.getenv("PATRA_AGENT_LLM_API_BASE", "http://127.0.0.1:1234/v1")


def get_llm_model() -> str:
    return os.getenv("PATRA_AGENT_LLM_MODEL", "qwen/qwen3.5-9b")


def get_llm_api_key() -> str:
    return os.getenv("PATRA_AGENT_LLM_API_KEY", "lm-studio")


def get_llm_timeout_seconds() -> int:
    return int(os.getenv("PATRA_AGENT_LLM_TIMEOUT_SECONDS", "60") or "60")


# ---------------------------------------------------------------------------
# Automated ingestion
# ---------------------------------------------------------------------------

def get_ingestion_min_file_bytes() -> int:
    return int(os.getenv("INGESTION_MIN_FILE_BYTES", "1024") or "1024")


def get_ingestion_max_file_bytes() -> int:
    return int(os.getenv("INGESTION_MAX_FILE_BYTES", str(500 * 1024 * 1024)) or str(500 * 1024 * 1024))


def get_ingestion_staging_dir() -> str:
    return os.getenv("INGESTION_STAGING_DIR", "/data/ingestion-staging")


def get_ingestion_llm_api_base() -> str:
    return os.getenv("INGESTION_LLM_API_BASE", get_llm_api_base())


def get_ingestion_llm_model() -> str:
    return os.getenv("INGESTION_LLM_MODEL", "").strip()


def get_ingestion_llm_api_key() -> str:
    return os.getenv("INGESTION_LLM_API_KEY", get_llm_api_key())


def is_ingestion_llm_enabled() -> bool:
    return os.getenv("INGESTION_LLM_ENABLED", "true").strip().lower() == "true"


def is_ingestion_llm_fallback_to_code() -> bool:
    return os.getenv("INGESTION_LLM_FALLBACK_TO_CODE", "true").strip().lower() == "true"


# ---------------------------------------------------------------------------
# Asset backups
# ---------------------------------------------------------------------------

def get_asset_backup_dir() -> str:
    return os.getenv("ASSET_BACKUP_DIR", "/tmp/patra-asset-backups")


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

def get_mcp_port() -> int:
    return int(os.getenv("MCP_PORT", "8050"))
