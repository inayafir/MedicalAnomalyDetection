from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class NotFoundError(Exception):
    def __init__(self, detail: str = "Resource not found"):
        self.detail = detail


class AggregationError(Exception):
    def __init__(self, detail: str = "Malformed ML output"):
        self.detail = detail


class StorageError(Exception):
    def __init__(self, detail: str = "File storage error"):
        self.detail = detail


async def not_found_handler(request: Request, exc: NotFoundError):
    return JSONResponse(status_code=404, content={"detail": exc.detail})


async def aggregation_error_handler(request: Request, exc: AggregationError):
    return JSONResponse(status_code=422, content={"detail": exc.detail})


async def storage_error_handler(request: Request, exc: StorageError):
    return JSONResponse(status_code=400, content={"detail": exc.detail})


async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
