from __future__ import annotations

from fastapi import APIRouter, Depends

from rest_server.deps import PatraActor, require_authenticated_actor
from rest_server.features.baseline_training.models import (
    BaselineTrainingStubRequest,
    BaselineTrainingStubResponse,
)
from rest_server.features.baseline_training.service import run_baseline_training_stub


router = APIRouter(prefix="/api/baseline-training", tags=["baseline-training"])


@router.post("/run-stub", response_model=BaselineTrainingStubResponse)
async def run_stub(
    payload: BaselineTrainingStubRequest,
    actor: PatraActor = Depends(require_authenticated_actor),
) -> BaselineTrainingStubResponse:
    return run_baseline_training_stub(
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
