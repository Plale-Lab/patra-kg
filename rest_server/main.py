from contextlib import asynccontextmanager
import asyncio
import logging
import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncpg

from rest_server.database import close_pool, get_pool, init_pool
from rest_server.routes import (
    agent_tools,
    ask_patra,
    assets,
    automated_ingestion,
    baseline_training,
    dataset_assembly,
    datasheets,
    experiments,
    intent_schema,
    metadata_discovery,
    mvp_demo_report,
    model_cards,
    submissions,
    tickets,
    training_readiness,
)

log = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting Patra FastAPI backend")
    pool = None
    db_startup_timeout = int(os.getenv("DB_STARTUP_TIMEOUT_SECONDS", "12") or "12")
    try:
        pool = await asyncio.wait_for(init_pool(), timeout=db_startup_timeout)
    except Exception:
        log.exception("Database initialization failed within startup timeout; starting in degraded mode")
    backup_task = None
    interval_seconds = int(os.getenv("ASSET_PERIODIC_BACKUP_INTERVAL_SECONDS", "0") or "0")
    if interval_seconds > 0 and pool is not None:
        async def _backup_loop():
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    await assets.run_periodic_backup_once(pool)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("Periodic asset backup run failed")

        backup_task = asyncio.create_task(_backup_loop(), name="patra-periodic-asset-backups")
    yield
    if backup_task is not None:
        backup_task.cancel()
        try:
            await backup_task
        except asyncio.CancelledError:
            pass
    log.info("Stopping Patra FastAPI backend")
    await close_pool()


app = FastAPI(
    title="Patra Privacy API",
    description="API for model cards and datasheets with JWT-aware privacy",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_cards.router)
app.include_router(datasheets.router)
app.include_router(assets.router)
app.include_router(tickets.router)
app.include_router(agent_tools.router)
app.include_router(submissions.router)

if _env_flag("ENABLE_ASK_PATRA", default=False):
    app.include_router(ask_patra.router)
    log.info("Ask Patra routes enabled")
else:
    log.info("Ask Patra routes disabled")

if _env_flag("ENABLE_INTENT_SCHEMA", default=False):
    app.include_router(intent_schema.router)
    log.info("Intent schema routes enabled")
else:
    log.info("Intent schema routes disabled")

if _env_flag("ENABLE_METADATA_DISCOVERY", default=False):
    app.include_router(metadata_discovery.router)
    log.info("Metadata discovery routes enabled")
else:
    log.info("Metadata discovery routes disabled")

if _env_flag("ENABLE_DATASET_ASSEMBLY", default=_env_flag("ENABLE_METADATA_DISCOVERY", default=False)):
    app.include_router(dataset_assembly.router)
    log.info("Dataset assembly routes enabled")
else:
    log.info("Dataset assembly routes disabled")

if _env_flag("ENABLE_TRAINING_READINESS", default=_env_flag("ENABLE_DATASET_ASSEMBLY", default=_env_flag("ENABLE_METADATA_DISCOVERY", default=False))):
    app.include_router(training_readiness.router)
    log.info("Training readiness routes enabled")
else:
    log.info("Training readiness routes disabled")

if _env_flag("ENABLE_BASELINE_TRAINING_STUB", default=_env_flag("ENABLE_TRAINING_READINESS", default=_env_flag("ENABLE_DATASET_ASSEMBLY", default=_env_flag("ENABLE_METADATA_DISCOVERY", default=False)))):
    app.include_router(baseline_training.router)
    log.info("Baseline training stub routes enabled")
else:
    log.info("Baseline training stub routes disabled")

if _env_flag("ENABLE_MVP_DEMO_REPORT", default=_env_flag("ENABLE_BASELINE_TRAINING_STUB", default=_env_flag("ENABLE_TRAINING_READINESS", default=_env_flag("ENABLE_DATASET_ASSEMBLY", default=_env_flag("ENABLE_METADATA_DISCOVERY", default=False))))):
    app.include_router(mvp_demo_report.router)
    log.info("MVP demo report routes enabled")
else:
    log.info("MVP demo report routes disabled")

if _env_flag("ENABLE_AUTOMATED_INGESTION", default=False):
    app.include_router(automated_ingestion.router)
    log.info("Automated ingestion routes enabled")
else:
    log.info("Automated ingestion routes disabled")

if _env_flag("ENABLE_DOMAIN_EXPERIMENTS", default=False):
    app.include_router(experiments.router)
    log.info("Experiment routes enabled")
else:
    log.info("Experiment routes disabled")


@app.get("/")
async def root():
    return {"message": "Welcome to the Patra Privacy API"}


@app.get("/healthz")
async def healthz():
    """Liveness probe: confirms the process is up."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(pool: asyncpg.Pool = Depends(get_pool)):
    """Readiness probe: confirms the API can still talk to PostgreSQL."""
    try:
        async with pool.acquire() as conn:
            value = await conn.fetchval("SELECT 1")
    except Exception as exc:
        log.exception("Readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail="database unavailable")
    if value != 1:
        raise HTTPException(status_code=503, detail="database unavailable")
    return {"status": "ok"}
