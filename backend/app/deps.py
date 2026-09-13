from __future__ import annotations

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(request: Request, api_key: str | None = Security(_api_key_header)):
    if not settings.API_KEY:
        return
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_api_key_dependency():
    if not settings.API_KEY:
        return None
    return require_api_key
