import asyncio
import json
from pathlib import Path as FilePath

import asyncpg
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from rest_server.agent_tool_models import (
    MissingColumnAnalysisRequest,
    MissingColumnAnalysisResponse,
    PaperSchemaSearchRequest,
    PaperSchemaSearchResponse,
    SchemaPoolItem,
    SynthesizeDatasetRequest,
    SynthesizeDatasetResponse,
)
from rest_server.database import get_pool
from rest_server.patra_agent_service import (
    AgentServiceError,
    _pair_map,
    _normalize_cache_dir,
    analyze_missing_columns_for_candidate,
    list_schema_pool,
    run_paper_schema_search,
    run_uploaded_paper_schema_search,
)
from rest_server.patra_synthesis_service import (
    SynthesisServiceError,
    generate_synthesized_dataset,
)
router = APIRouter(prefix="/agent-tools", tags=["agent-tools"])


async def _insert_generated_artifact(conn: asyncpg.Connection, payload: dict[str, object]) -> None:
    await conn.execute(
        """
        INSERT INTO generated_dataset_artifacts (
            artifact_key, title, source_dataset_id, submitted_by, planner_mode,
            query_schema, generated_schema, derivation_plan, validation_report,
            metadata, output_csv_path, output_schema_path, status, updated_at
        )
        VALUES (
            $1, $2, $3, $4, $5,
            $6::jsonb, $7::jsonb, $8::jsonb, $9::jsonb,
            $10::jsonb, $11, $12, 'generated', NOW()
        )
        ON CONFLICT (artifact_key)
        DO UPDATE SET
            title = EXCLUDED.title,
            source_dataset_id = EXCLUDED.source_dataset_id,
            submitted_by = EXCLUDED.submitted_by,
            planner_mode = EXCLUDED.planner_mode,
            query_schema = EXCLUDED.query_schema,
            generated_schema = EXCLUDED.generated_schema,
            derivation_plan = EXCLUDED.derivation_plan,
            validation_report = EXCLUDED.validation_report,
            metadata = EXCLUDED.metadata,
            output_csv_path = EXCLUDED.output_csv_path,
            output_schema_path = EXCLUDED.output_schema_path,
            updated_at = NOW()
        """,
        payload["artifact_key"],
        payload["title"],
        payload["source_dataset_id"],
        payload["submitted_by"],
        payload["planner_mode"],
        json.dumps(payload["query_schema"]),
        json.dumps(payload["generated_schema"]),
        json.dumps(payload["derivation_plan"]),
        json.dumps(payload["validation_report"]),
        json.dumps(payload["metadata"]),
        payload["output_csv_path"],
        payload["output_schema_path"],
    )


async def _get_generated_artifact(conn: asyncpg.Connection, artifact_key: str):
    row = await conn.fetchrow(
        """
        SELECT artifact_key, title, source_dataset_id, submitted_by, planner_mode,
               query_schema, generated_schema, derivation_plan, validation_report,
               metadata, output_csv_path, output_schema_path, status
        FROM generated_dataset_artifacts
        WHERE artifact_key = $1
        """,
        artifact_key,
    )
    if not row:
        return None
    payload = dict(row)
    for key in ("query_schema", "generated_schema", "derivation_plan", "validation_report", "metadata"):
        if isinstance(payload.get(key), str):
            payload[key] = json.loads(payload[key])
    return payload


@router.get("/schema-pool", response_model=list[SchemaPoolItem])
async def get_schema_pool():
    try:
        return await asyncio.to_thread(list_schema_pool)
    except AgentServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/paper-schema-search", response_model=PaperSchemaSearchResponse)
async def paper_schema_search(payload: PaperSchemaSearchRequest):
    try:
        return await asyncio.to_thread(
            run_paper_schema_search,
            None,
            payload.document_url,
            payload.document_text,
            payload.document_format,
            payload.top_k,
            payload.disable_llm,
            payload.api_base,
            payload.model,
            payload.api_key,
            payload.timeout_seconds,
            payload.cache_dir,
        )
    except AgentServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/paper-schema-search-upload", response_model=PaperSchemaSearchResponse)
async def paper_schema_search_upload(
    file: UploadFile = File(...),
    document_format: str | None = Form(default=None),
    top_k: int = Form(default=5),
    disable_llm: bool = Form(default=True),
    api_base: str | None = Form(default=None),
    model: str | None = Form(default=None),
    api_key: str | None = Form(default=None),
    timeout_seconds: int = Form(default=60),
    cache_dir: str | None = Form(default=None),
):
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return await asyncio.to_thread(
            run_uploaded_paper_schema_search,
            file_bytes,
            file.filename,
            document_format,
            top_k,
            disable_llm,
            api_base,
            model,
            api_key,
            timeout_seconds,
            cache_dir,
        )
    except AgentServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        await file.close()


@router.post("/missing-column-analysis", response_model=MissingColumnAnalysisResponse)
async def missing_column_analysis(payload: MissingColumnAnalysisRequest):
    try:
        return await asyncio.to_thread(
            analyze_missing_columns_for_candidate,
            payload.query_schema,
            payload.candidate_dataset_id,
            payload.cache_dir,
        )
    except AgentServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/generate-synthesized-dataset", response_model=SynthesizeDatasetResponse)
async def generate_synthesized_dataset_route(
    payload: SynthesizeDatasetRequest,
    pool: asyncpg.Pool = Depends(get_pool),
):
    try:
        result = await asyncio.to_thread(
            generate_synthesized_dataset,
            payload.query_schema,
            payload.candidate_dataset_id,
            payload.selected_fields,
            payload.use_llm_plan,
            payload.submitted_by,
            payload.api_base,
            payload.model,
            payload.api_key,
            payload.timeout_seconds,
            payload.cache_dir,
        )
    except (AgentServiceError, SynthesisServiceError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    storage = result.pop("storage")
    artifact = result["artifact"]
    async with pool.acquire() as conn:
        await _insert_generated_artifact(
            conn,
            {
                "artifact_key": artifact["artifact_key"],
                "title": artifact["title"],
                "source_dataset_id": artifact["source_dataset_id"],
                "submitted_by": payload.submitted_by,
                "planner_mode": artifact["planner_mode"],
                "query_schema": storage["query_schema"],
                "generated_schema": storage["generated_schema"],
                "derivation_plan": storage["derivation_plan"],
                "validation_report": storage["validation_report"],
                "metadata": storage["metadata"],
                "output_csv_path": storage["output_csv_path"],
                "output_schema_path": storage["output_schema_path"],
            },
        )
    return result


@router.get("/generated-artifacts/{artifact_key}")
async def get_generated_artifact(artifact_key: str, pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        row = await _get_generated_artifact(conn, artifact_key)
    if not row:
        raise HTTPException(status_code=404, detail="Generated artifact not found")
    return {
        "artifact_key": row["artifact_key"],
        "title": row["title"],
        "source_dataset_id": row["source_dataset_id"],
        "submitted_by": row["submitted_by"],
        "planner_mode": row["planner_mode"],
        "status": row["status"],
        "metadata": row["metadata"],
        "output_csv_download_url": f"/agent-tools/generated-artifacts/{artifact_key}/download.csv",
        "output_schema_download_url": f"/agent-tools/generated-artifacts/{artifact_key}/download-schema",
    }


@router.get("/generated-artifacts/{artifact_key}/download.csv")
async def download_generated_csv(artifact_key: str, pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        row = await _get_generated_artifact(conn, artifact_key)
    if not row:
        raise HTTPException(status_code=404, detail="Generated artifact not found")
    path = FilePath(row["output_csv_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Generated CSV file not found")
    return FileResponse(path, media_type="text/csv", filename=f"{artifact_key}.csv")


@router.get("/generated-artifacts/{artifact_key}/download-schema")
async def download_generated_schema(artifact_key: str, pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        row = await _get_generated_artifact(conn, artifact_key)
    if not row:
        raise HTTPException(status_code=404, detail="Generated artifact not found")
    path = FilePath(row["output_schema_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Generated schema file not found")
    return FileResponse(path, media_type="application/json", filename=f"{artifact_key}-schema.json")


