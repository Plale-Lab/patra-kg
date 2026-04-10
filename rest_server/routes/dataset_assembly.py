from __future__ import annotations

from fastapi import APIRouter, Depends

from rest_server.deps import PatraActor, require_authenticated_actor
from rest_server.features.dataset_assembly.models import (
    DatasetAssemblyPlanRequest,
    DatasetAssemblyPlanResponse,
    DatasetCompositionPreviewRequest,
    DatasetCompositionPreviewResponse,
)
from rest_server.features.dataset_assembly.service import (
    build_dataset_assembly_plan,
    build_dataset_composition_preview,
)


router = APIRouter(prefix="/api/dataset-assembly", tags=["dataset-assembly"])


@router.post("/plan", response_model=DatasetAssemblyPlanResponse)
async def build_plan(
    payload: DatasetAssemblyPlanRequest,
    actor: PatraActor = Depends(require_authenticated_actor),
) -> DatasetAssemblyPlanResponse:
    return build_dataset_assembly_plan(
        intent_schema=payload.intent_schema,
        top_k=payload.top_k,
        disable_llm=payload.disable_llm,
        api_base=payload.api_base,
        model=payload.model,
        api_key=payload.api_key,
        timeout_seconds=payload.timeout_seconds,
        cache_dir=payload.cache_dir,
    )


@router.post("/compose-preview", response_model=DatasetCompositionPreviewResponse)
async def compose_preview(
    payload: DatasetCompositionPreviewRequest,
    actor: PatraActor = Depends(require_authenticated_actor),
) -> DatasetCompositionPreviewResponse:
    return build_dataset_composition_preview(
        intent_schema=payload.intent_schema,
        top_k=payload.top_k,
        disable_llm=payload.disable_llm,
        api_base=payload.api_base,
        model=payload.model,
        api_key=payload.api_key,
        timeout_seconds=payload.timeout_seconds,
        cache_dir=payload.cache_dir,
        preview_row_limit=payload.preview_row_limit,
    )
