import logging

from fastapi import Request

log = logging.getLogger(__name__)

TAPIS_TOKEN_HEADER = "X-Tapis-Token"


def get_include_private(request: Request) -> bool:
    """Return True when the caller presents a Tapis token via X-Tapis-Token.

    The patra-toolkit authenticates against Tapis OAuth2 and passes the
    resulting access token in the ``X-Tapis-Token`` header.  Presence of
    any non-empty value is treated as authenticated (matching the legacy
    Flask server behaviour).  No token falls back to public-only.
    """
    token = request.headers.get(TAPIS_TOKEN_HEADER)
    if not token:
        return False
    log.debug("X-Tapis-Token present – including private records")
    return True
