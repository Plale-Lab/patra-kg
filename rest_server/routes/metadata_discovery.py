from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from rest_server.deps import PatraActor, require_authenticated_actor
from rest_server.features.metadata_discovery.column_profile import profile_csv_bytes
from rest_server.features.metadata_discovery.models import (
    ColumnProfileResponse,
    InternalAssetIndexRefreshResponse,
    InternalAssetIndexStatusResponse,
    MetadataDiscoveryRequest,
    MetadataDiscoveryResponse,
    UploadedCsvProfileResponse,
)
from rest_server.features.metadata_discovery.service import (
    discover_metadata,
    internal_asset_index_status,
    refresh_internal_asset_index,
)


router = APIRouter(prefix="/api/metadata-discovery", tags=["metadata-discovery"])


@router.get("/internal-index/status", response_model=InternalAssetIndexStatusResponse)
async def internal_index_status(
    actor: PatraActor = Depends(require_authenticated_actor),
) -> InternalAssetIndexStatusResponse:
    return InternalAssetIndexStatusResponse(**internal_asset_index_status(cache_dir=None))


@router.post("/internal-index/refresh", response_model=InternalAssetIndexRefreshResponse)
async def internal_index_refresh(
    actor: PatraActor = Depends(require_authenticated_actor),
) -> InternalAssetIndexRefreshResponse:
    return InternalAssetIndexRefreshResponse(**refresh_internal_asset_index(cache_dir=None))


@router.post("/discover", response_model=MetadataDiscoveryResponse)
async def discover(
    payload: MetadataDiscoveryRequest,
    actor: PatraActor = Depends(require_authenticated_actor),
) -> MetadataDiscoveryResponse:
    return discover_metadata(
        intent_schema=payload.intent_schema,
        top_k=payload.top_k,
        disable_llm=payload.disable_llm,
        api_base=payload.api_base,
        model=payload.model,
        api_key=payload.api_key,
        timeout_seconds=payload.timeout_seconds,
        cache_dir=payload.cache_dir,
    )


@router.post("/profile-upload", response_model=UploadedCsvProfileResponse)
async def profile_upload(
    file: UploadFile = File(...),
    sample_limit: int = Form(default=250),
    actor: PatraActor = Depends(require_authenticated_actor),
) -> UploadedCsvProfileResponse:
    payload = await file.read()
    await file.close()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
    try:
        profiled = profile_csv_bytes(
            payload,
            source_label=file.filename or "uploaded.csv",
            sample_limit=max(1, min(sample_limit, 1000)),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to profile uploaded CSV: {exc}") from exc

    return UploadedCsvProfileResponse(
        status="ok",
        message="Deterministic column profiling completed.",
        source_label=profiled["source_label"],
        column_count=profiled["column_count"],
        sample_row_count=profiled["sample_row_count"],
        columns=[ColumnProfileResponse(**column) for column in profiled["column_profiles"]],
        schema_contract={
            "type": "object",
            "properties": profiled["schema_properties"],
            "required": profiled["required"],
            "additionalProperties": False,
        },
    )
