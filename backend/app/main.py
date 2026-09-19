from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings, validate_production_config
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
from app.middleware import APIKeyMiddleware, RequestLoggingMiddleware, SecurityHeadersMiddleware
from app.ml_interface import load_models
from app.rate_limit import limiter
from app.routers import health, images, patients, predictions, reports, files

# --- Logging ---
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_LEVEL = logging.DEBUG if not settings.is_production else logging.INFO

if settings.is_production:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
else:
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, force=True)

logger = logging.getLogger(__name__)

# --- Sentry (optional) ---
if settings.SENTRY_DSN:
    try:
        import sentry_sdk
        sentry_sdk.init(dsn=settings.SENTRY_DSN, environment=settings.ENVIRONMENT)
        logger.info("Sentry initialized for environment=%s", settings.ENVIRONMENT)
    except Exception:
        logger.warning("Failed to initialize Sentry — continuing without error tracking")


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_production_config()
    Path(settings.STORAGE_ROOT).mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    load_models()
    yield


app = FastAPI(
    title="Chest X-Ray Anomaly Detection API",
    description="Multi-class chest X-ray classification + detection using ResNet-50 and YOLOv8m.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# --- Middleware (order matters: outermost runs first) ---
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rate limiting ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Exception handlers ---
app.add_exception_handler(NotFoundError, not_found_handler)
app.add_exception_handler(AggregationError, aggregation_error_handler)
app.add_exception_handler(StorageError, storage_error_handler)
app.add_exception_handler(Exception, global_exception_handler)

# --- Routers ---
app.include_router(health.router)
app.include_router(images.router)
app.include_router(predictions.router)
app.include_router(reports.router)
app.include_router(patients.router)
app.include_router(files.router)
