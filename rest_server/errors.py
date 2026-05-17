from fastapi import HTTPException


ASSET_NOT_AVAILABLE_DETAIL = "assets not available or not visible."


def asset_not_available_or_visible() -> HTTPException:
    return HTTPException(status_code=404, detail=ASSET_NOT_AVAILABLE_DETAIL)


def not_found(resource: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{resource} not found")


def auth_required() -> HTTPException:
    return HTTPException(status_code=401, detail="Authentication required")


def admin_required() -> HTTPException:
    return HTTPException(status_code=403, detail="Admin privileges required")


def database_unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail="database unavailable")


def service_not_configured(service: str) -> HTTPException:
    return HTTPException(status_code=503, detail=f"{service} is not configured")
