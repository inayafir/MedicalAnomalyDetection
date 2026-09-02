from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import Base, engine
from app.exceptions import (
    AggregationError,
    NotFoundError,
    StorageError,
    aggregation_error_handler,
    global_exception_handler,
    not_found_handler,
    storage_error_handler,
)
from app.ml_interface import load_models
from app.routers import health, images, patients, predictions, reports, files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Path(settings.STORAGE_ROOT).mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    load_models()
    yield
    # Shutdown (nothing to clean up)


app = FastAPI(
    title="Chest X-Ray Anomaly Detection API",
    lifespan=lifespan,
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Exception handlers ---
app.add_exception_handler(NotFoundError, not_found_handler)
app.add_exception_handler(AggregationError, aggregation_error_handler)
app.add_exception_handler(StorageError, storage_error_handler)
app.add_exception_handler(Exception, global_exception_handler)

# --- Request logging middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    logger.info(
        "%s %s -> %s (%.3fs)",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response

# --- Routers ---
app.include_router(health.router)
app.include_router(images.router)
app.include_router(predictions.router)
app.include_router(reports.router)
app.include_router(patients.router)
app.include_router(files.router)
