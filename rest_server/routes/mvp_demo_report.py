from __future__ import annotations

from fastapi import APIRouter, Depends

from rest_server.deps import PatraActor, require_authenticated_actor
from rest_server.features.mvp_demo_report.models import MvpDemoReportRequest, MvpDemoReportResponse
from rest_server.features.mvp_demo_report.service import build_mvp_demo_report


router = APIRouter(prefix="/api/mvp-demo-report", tags=["mvp-demo-report"])


@router.post("/generate", response_model=MvpDemoReportResponse)
async def generate(
    payload: MvpDemoReportRequest,
    actor: PatraActor = Depends(require_authenticated_actor),
) -> MvpDemoReportResponse:
    return build_mvp_demo_report(
        intent_text=payload.intent_text,
        context=payload.context,
        max_fields=payload.max_fields,
        top_k=payload.top_k,
        disable_llm=payload.disable_llm,
        preview_row_limit=payload.preview_row_limit,
    )
