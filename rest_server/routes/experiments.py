"""Experiment data endpoints for supported Patra knowledge domains."""

from fastapi import APIRouter, Depends, HTTPException, Path, Query
import asyncpg

from rest_server.database import get_pool
from rest_server.errors import not_found
from rest_server.models import (
    DeploymentDetail,
    ExperimentDetail,
    ExperimentImage,
    ExperimentListItem,
    ExperimentSummary,
    ExperimentUser,
)
from shared.constants import DOMAIN_TABLES

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _resolve_tables(domain: str) -> tuple[str, str]:
    tables = DOMAIN_TABLES.get(domain)
    if not tables:
        raise HTTPException(status_code=404, detail=f"Unknown domain: {domain}")
    return tables["events"], tables["power"]


def _float(value):
    return float(value) if value is not None else None


@router.get("/{domain}/users", response_model=list[ExperimentUser])
async def list_experiment_users(
    domain: str = Path(...),
    pool: asyncpg.Pool = Depends(get_pool),
):
    events_table, _ = _resolve_tables(domain)
    query = f"""
        SELECT DISTINCT user_id, user_id AS username
        FROM {events_table}
        ORDER BY user_id
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
    return [ExperimentUser(user_id=row["user_id"], username=row["username"]) for row in rows]


@router.get("/{domain}/users/{user_id}/summary", response_model=list[ExperimentSummary])
async def get_user_experiment_summary(
    domain: str = Path(...),
    user_id: str = Path(...),
    pool: asyncpg.Pool = Depends(get_pool),
):
    events_table, _ = _resolve_tables(domain)
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

    return [
        ExperimentSummary(
            experiment_id=row["experiment_id"],
            user_id=row["user_id"],
            model_id=row["model_id"],
            device_id=row["device_id"],
            start_at=row["start_at"].isoformat() if row["start_at"] else None,
            total_images=row["total_images"],
            saved_images=int(row["saved_images"]),
            precision=_float(row["precision"]),
            recall=_float(row["recall"]),
            f1_score=_float(row["f1_score"]),
        )
        for row in rows
    ]


@router.get("/{domain}/users/{user_id}/list", response_model=list[ExperimentListItem])
async def list_user_experiments(
    domain: str = Path(...),
    user_id: str = Path(...),
    pool: asyncpg.Pool = Depends(get_pool),
):
    events_table, _ = _resolve_tables(domain)
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
            experiment_id=row["experiment_id"],
            start_at=row["start_at"].isoformat() if row["start_at"] else None,
            device_id=row["device_id"],
            model_id=row["model_id"],
        )
        for row in rows
    ]


@router.get("/{domain}/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment_detail(
    domain: str = Path(...),
    experiment_id: str = Path(...),
    pool: asyncpg.Pool = Depends(get_pool),
):
    events_table, _ = _resolve_tables(domain)
    query = f"""
        SELECT *
        FROM {events_table}
        WHERE experiment_id = $1
        ORDER BY image_count DESC
        LIMIT 1
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, experiment_id)
    if not row:
        raise not_found("Experiment")

    return ExperimentDetail(
        experiment_id=row["experiment_id"],
        model_id=row["model_id"],
        device_id=row["device_id"],
        start_at=row["image_receiving_timestamp"].isoformat() if row["image_receiving_timestamp"] else None,
        total_images=row["total_images"],
        total_predictions=row["total_predictions"],
        total_ground_truth_objects=row["total_ground_truth_objects"],
        true_positives=row["true_positives"],
        false_positives=row["false_positives"],
        false_negatives=row["false_negatives"],
        precision=_float(row["precision"]),
        recall=_float(row["recall"]),
        f1_score=_float(row["f1_score"]),
        map_50=_float(row["map_50"]),
        map_50_95=_float(row["map_50_95"]),
        mean_iou=_float(row["mean_iou"]),
    )


@router.get("/{domain}/{experiment_id}/images", response_model=list[ExperimentImage])
async def get_experiment_images(
    domain: str = Path(...),
    experiment_id: str = Path(...),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    pool: asyncpg.Pool = Depends(get_pool),
):
    events_table, _ = _resolve_tables(domain)
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
            image_name=row["image_name"],
            ground_truth=row["ground_truth"],
            label=row["label"],
            probability=_float(row["probability"]),
            image_decision=row["image_decision"],
            flattened_scores=row["flattened_scores"],
            image_receiving_timestamp=row["image_receiving_timestamp"].isoformat() if row["image_receiving_timestamp"] else None,
            image_scoring_timestamp=row["image_scoring_timestamp"].isoformat() if row["image_scoring_timestamp"] else None,
        )
        for row in rows
    ]


@router.get("/{domain}/{experiment_id}/power", response_model=DeploymentDetail | None)
async def get_experiment_power(
    domain: str = Path(...),
    experiment_id: str = Path(...),
    pool: asyncpg.Pool = Depends(get_pool),
):
    _, power_table = _resolve_tables(domain)
    query = f"SELECT * FROM {power_table} WHERE experiment_id = $1"
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, experiment_id)
    if not row:
        return None

    return DeploymentDetail(
        experiment_id=row["experiment_id"],
        image_generating_plugin_cpu_power_consumption=_float(row["image_generating_plugin_cpu_power_consumption"]),
        image_generating_plugin_gpu_power_consumption=_float(row["image_generating_plugin_gpu_power_consumption"]),
        power_monitor_plugin_cpu_power_consumption=_float(row["power_monitor_plugin_cpu_power_consumption"]),
        power_monitor_plugin_gpu_power_consumption=_float(row["power_monitor_plugin_gpu_power_consumption"]),
        image_scoring_plugin_cpu_power_consumption=_float(row["image_scoring_plugin_cpu_power_consumption"]),
        image_scoring_plugin_gpu_power_consumption=_float(row["image_scoring_plugin_gpu_power_consumption"]),
        total_cpu_power_consumption=_float(row["total_cpu_power_consumption"]),
        total_gpu_power_consumption=_float(row["total_gpu_power_consumption"]),
    )
