from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class NotFoundError(Exception):
    def __init__(self, detail: str = "Resource not found"):
        self.detail = detail


class AggregationError(Exception):
    def __init__(self, detail: str = "Malformed ML output"):
        self.detail = detail


async def not_found_handler(request: Request, exc: NotFoundError):
    return JSONResponse(status_code=404, content={"detail": exc.detail})


async def aggregation_error_handler(request: Request, exc: AggregationError):
    return JSONResponse(status_code=422, content={"detail": exc.detail})
