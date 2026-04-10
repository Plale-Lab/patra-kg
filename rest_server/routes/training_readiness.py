from __future__ import annotations

from fastapi import APIRouter, Depends

from rest_server.deps import PatraActor, require_authenticated_actor
from rest_server.features.training_readiness.models import TrainingReadinessRequest, TrainingReadinessResponse
from rest_server.features.training_readiness.service import evaluate_training_readiness


router = APIRouter(prefix="/api/training-readiness", tags=["training-readiness"])


@router.post("/evaluate", response_model=TrainingReadinessResponse)
async def evaluate(
    payload: TrainingReadinessRequest,
    actor: PatraActor = Depends(require_authenticated_actor),
) -> TrainingReadinessResponse:
    return evaluate_training_readiness(
        intent_schema=payload.intent_schema,
        top_k=payload.top_k,
        disable_llm=payload.disable_llm,
        api_base=payload.api_base,
        model=payload.model,
        api_key=payload.api_key,
        timeout_seconds=payload.timeout_seconds,
        cache_dir=payload.cache_dir,
    )
