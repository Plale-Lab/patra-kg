from contextlib import asynccontextmanager
import asyncio
import logging
import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncpg

from rest_server.database import close_pool, get_pool, init_pool
from rest_server.routes import agent_tools, ask_patra, assets, automated_ingestion, datasheets, model_cards, submissions, tickets

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting Patra FastAPI backend")
    pool = await init_pool()
    backup_task = None
    interval_seconds = int(os.getenv("ASSET_PERIODIC_BACKUP_INTERVAL_SECONDS", "0") or "0")
    if interval_seconds > 0:
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
app.include_router(submissions.router)
app.include_router(tickets.router)
app.include_router(agent_tools.router)
app.include_router(automated_ingestion.router)
app.include_router(ask_patra.router)


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
