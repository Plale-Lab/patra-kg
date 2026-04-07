"""Experiments endpoints — flat table queries, parameterised by domain."""

from fastapi import APIRouter, Depends, HTTPException, Path, Query
import asyncpg

from rest_server.database import get_pool
from rest_server.models import (
    DeploymentDetail,
    ExperimentDetail,
    ExperimentImage,
    ExperimentListItem,
    ExperimentSummary,
    ExperimentUser,
)

router = APIRouter(prefix="/experiments", tags=["experiments"])

# Domain → table mapping (safe allowlist — never user-supplied)
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


def _tables(domain: str):
    """Return (events_table, power_table) or raise 404."""
    tables = DOMAIN_TABLES.get(domain)
    if not tables:
        raise HTTPException(status_code=404, detail=f"Unknown domain: {domain}")
    return tables["events"], tables["power"]


@router.get("/{domain}/users", response_model=list[ExperimentUser])
async def list_experiment_users(
    domain: str = Path(...),
    pool: asyncpg.Pool = Depends(get_pool),
):
    """List distinct users that have at least one experiment event."""
    events_table, _ = _tables(domain)
    query = f"""
        SELECT DISTINCT user_id, user_id AS username
        FROM {events_table}
        ORDER BY user_id
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
    return [ExperimentUser(user_id=r["user_id"], username=r["username"]) for r in rows]


@router.get("/{domain}/users/{user_id}/summary", response_model=list[ExperimentSummary])
async def get_user_experiment_summary(
    domain: str = Path(...),
    user_id: str = Path(...),
    pool: asyncpg.Pool = Depends(get_pool),
):
    """Experiment summary table for a given user."""
    events_table, _ = _tables(domain)
    query = f"""
        SELECT
            experiment_id,
            user_id,
            model_id,
            device_id,
            MIN(image_receiving_timestamp) AS start_at,
            MAX(total_images) AS total_images,
            SUM(CASE WHEN image_decision = 'Save' THEN 1 ELSE 0 END) AS saved_images,
            MAX(precision) AS precision,
            MAX(recall) AS recall,
            MAX(f1_score) AS f1_score
        FROM {events_table}
        WHERE user_id = $1
        GROUP BY experiment_id, user_id, model_id, device_id
        ORDER BY MIN(image_receiving_timestamp) DESC
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, user_id)

    def _float(v):
        return float(v) if v is not None else None

    return [
        ExperimentSummary(
            experiment_id=r["experiment_id"],
            user_id=r["user_id"],
            model_id=r["model_id"],
            device_id=r["device_id"],
            start_at=r["start_at"].isoformat() if r["start_at"] else None,
            total_images=r["total_images"],
            saved_images=int(r["saved_images"]),
            precision=_float(r["precision"]),
            recall=_float(r["recall"]),
            f1_score=_float(r["f1_score"]),
        )
        for r in rows
    ]


@router.get("/{domain}/users/{user_id}/list", response_model=list[ExperimentListItem])
async def list_user_experiments(
    domain: str = Path(...),
    user_id: str = Path(...),
    pool: asyncpg.Pool = Depends(get_pool),
):
    """Experiment IDs for the dropdown selector."""
    events_table, _ = _tables(domain)
    query = f"""
        SELECT DISTINCT
            experiment_id,
            MIN(image_receiving_timestamp) AS start_at,
            device_id,
            model_id
        FROM {events_table}
        WHERE user_id = $1
        GROUP BY experiment_id, device_id, model_id
        ORDER BY MIN(image_receiving_timestamp) DESC
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, user_id)
    return [
        ExperimentListItem(
            experiment_id=r["experiment_id"],
            start_at=r["start_at"].isoformat() if r["start_at"] else None,
            device_id=r["device_id"],
            model_id=r["model_id"],
        )
        for r in rows
    ]


@router.get("/{domain}/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment_detail(
    domain: str = Path(...),
    experiment_id: str = Path(...),
    pool: asyncpg.Pool = Depends(get_pool),
):
    """Full experiment detail — latest metrics snapshot."""
    events_table, _ = _tables(domain)
    query = f"""
        SELECT *
        FROM {events_table}
        WHERE experiment_id = $1
        ORDER BY image_count DESC
        LIMIT 1
    """
    async with pool.acquire() as conn:
        r = await conn.fetchrow(query, experiment_id)
    if not r:
        raise HTTPException(status_code=404, detail="Experiment not found")

    def _float(v):
        return float(v) if v is not None else None

    return ExperimentDetail(
        experiment_id=r["experiment_id"],
        model_id=r["model_id"],
        device_id=r["device_id"],
        start_at=r["image_receiving_timestamp"].isoformat() if r["image_receiving_timestamp"] else None,
        total_images=r["total_images"],
        total_predictions=r["total_predictions"],
        total_ground_truth_objects=r["total_ground_truth_objects"],
        true_positives=r["true_positives"],
        false_positives=r["false_positives"],
        false_negatives=r["false_negatives"],
        precision=_float(r["precision"]),
        recall=_float(r["recall"]),
        f1_score=_float(r["f1_score"]),
        map_50=_float(r["map_50"]),
        map_50_95=_float(r["map_50_95"]),
        mean_iou=_float(r["mean_iou"]),
    )


@router.get("/{domain}/{experiment_id}/images", response_model=list[ExperimentImage])
async def get_experiment_images(
    domain: str = Path(...),
    experiment_id: str = Path(...),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    pool: asyncpg.Pool = Depends(get_pool),
):
    """Raw image data table (paginated)."""
    events_table, _ = _tables(domain)
    query = f"""
        SELECT
            image_name, ground_truth, label, probability,
            image_decision, flattened_scores,
            image_receiving_timestamp, image_scoring_timestamp
        FROM {events_table}
        WHERE experiment_id = $1
        ORDER BY image_receiving_timestamp ASC
        LIMIT $2 OFFSET $3
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, experiment_id, limit, skip)
    return [
        ExperimentImage(
            image_name=r["image_name"],
            ground_truth=r["ground_truth"],
            label=r["label"],
            probability=float(r["probability"]) if r["probability"] is not None else None,
            image_decision=r["image_decision"],
            flattened_scores=r["flattened_scores"],
            image_receiving_timestamp=r["image_receiving_timestamp"].isoformat() if r["image_receiving_timestamp"] else None,
            image_scoring_timestamp=r["image_scoring_timestamp"].isoformat() if r["image_scoring_timestamp"] else None,
        )
        for r in rows
    ]


@router.get("/{domain}/{experiment_id}/power", response_model=DeploymentDetail | None)
async def get_experiment_power(
    domain: str = Path(...),
    experiment_id: str = Path(...),
    pool: asyncpg.Pool = Depends(get_pool),
):
    """Power consumption breakdown for an experiment."""
    _, power_table = _tables(domain)
    query = f"SELECT * FROM {power_table} WHERE experiment_id = $1"
    async with pool.acquire() as conn:
        r = await conn.fetchrow(query, experiment_id)
    if not r:
        return None

    def _float(v):
        return float(v) if v is not None else None

    return DeploymentDetail(
        experiment_id=r["experiment_id"],
        image_generating_plugin_cpu_power_consumption=_float(r["image_generating_plugin_cpu_power_consumption"]),
        image_generating_plugin_gpu_power_consumption=_float(r["image_generating_plugin_gpu_power_consumption"]),
        power_monitor_plugin_cpu_power_consumption=_float(r["power_monitor_plugin_cpu_power_consumption"]),
        power_monitor_plugin_gpu_power_consumption=_float(r["power_monitor_plugin_gpu_power_consumption"]),
        image_scoring_plugin_cpu_power_consumption=_float(r["image_scoring_plugin_cpu_power_consumption"]),
        image_scoring_plugin_gpu_power_consumption=_float(r["image_scoring_plugin_gpu_power_consumption"]),
        total_cpu_power_consumption=_float(r["total_cpu_power_consumption"]),
        total_gpu_power_consumption=_float(r["total_gpu_power_consumption"]),
    )
