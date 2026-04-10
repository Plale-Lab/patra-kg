from __future__ import annotations

import asyncpg
from fastapi import APIRouter, Depends, Request

from rest_server.deps import get_request_actor
from rest_server.database import get_pool
from rest_server.features.ask_patra.models import (
    AskPatraBootstrapResponse,
    AskPatraChatRequest,
    AskPatraChatResponse,
    AskPatraExecuteRequest,
    AskPatraExecuteResponse,
)
from rest_server.features.ask_patra.service import _provider_label, answer_question, ensure_ask_patra_storage, execute_tool_action
from rest_server.features.ask_patra.tool_registry import get_tool_capabilities


router = APIRouter(prefix="/api/ask-patra", tags=["ask-patra"])


@router.get("/bootstrap", response_model=AskPatraBootstrapResponse)
async def ask_patra_bootstrap(actor=Depends(get_request_actor)):
    starters = ensure_ask_patra_storage()
    return AskPatraBootstrapResponse(
        enabled=True,
        provider=_provider_label(),
        starter_prompts=starters,
        tool_capabilities=get_tool_capabilities(actor),
    )


@router.post("/chat", response_model=AskPatraChatResponse)
async def ask_patra_chat(
    payload: AskPatraChatRequest,
    request: Request,
    actor=Depends(get_request_actor),
    pool: asyncpg.Pool = Depends(get_pool),
):
    async with pool.acquire() as conn:
        conversation_id, answer, model_used, citations, messages, starters, intent, tool_cards, suggested_actions, handoff, execution = await answer_question(
            conn,
            actor=actor,
            message=payload.message,
            conversation_id=payload.conversation_id,
            reset=payload.reset,
            request_tapis_token=(request.headers.get("X-Tapis-Token") or "").strip() or None,
        )
    return AskPatraChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        mode="llm" if model_used else "code_fallback",
        provider=_provider_label(),
        model_used=model_used,
        citations=citations,
        messages=messages,
        starter_prompts=starters,
        intent=intent,
        tool_cards=tool_cards,
        suggested_actions=suggested_actions,
        handoff=handoff,
        execution=execution,
    )


@router.post("/execute", response_model=AskPatraExecuteResponse)
async def ask_patra_execute(
    payload: AskPatraExecuteRequest,
    actor=Depends(get_request_actor),
    pool: asyncpg.Pool = Depends(get_pool),
):
    conversation_id, messages, execution = await execute_tool_action(
        actor=actor,
        tool_id=payload.tool_id,
        message=payload.message,
        conversation_id=payload.conversation_id,
        pool=pool,
        query=payload.query,
        prefilled_payload=payload.prefilled_payload,
        disable_llm=payload.disable_llm,
    )
    return AskPatraExecuteResponse(
        conversation_id=conversation_id,
        messages=messages,
        execution=execution,
    )
