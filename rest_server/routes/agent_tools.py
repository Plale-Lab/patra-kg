import asyncio

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from rest_server.agent_tool_models import (
    MissingColumnAnalysisRequest,
    MissingColumnAnalysisResponse,
    PaperSchemaSearchRequest,
    PaperSchemaSearchResponse,
    SchemaPoolItem,
    SynthesizeDatasetRequest,
    SynthesizeDatasetResponse,
)
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
async def generate_synthesized_dataset_route(payload: SynthesizeDatasetRequest):
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

    result.pop("storage", None)
    return result
