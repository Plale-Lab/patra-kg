from __future__ import annotations

from fastapi import APIRouter, Depends

from rest_server.deps import PatraActor, require_authenticated_actor
from rest_server.features.intent_schema.models import (
    IntentSchemaBootstrapResponse,
    IntentSchemaRequest,
    IntentSchemaResponse,
)
from rest_server.features.intent_schema.service import bootstrap_payload, generate_schema


router = APIRouter(prefix="/api/intent-schema", tags=["intent-schema"])


@router.get("/bootstrap", response_model=IntentSchemaBootstrapResponse)
async def bootstrap(actor: PatraActor = Depends(require_authenticated_actor)) -> IntentSchemaBootstrapResponse:
    return bootstrap_payload()


@router.post("/generate", response_model=IntentSchemaResponse)
async def generate(
    payload: IntentSchemaRequest,
    actor: PatraActor = Depends(require_authenticated_actor),
) -> IntentSchemaResponse:
    return generate_schema(
        intent_text=payload.intent_text,
        context=payload.context,
        max_fields=payload.max_fields,
    )
