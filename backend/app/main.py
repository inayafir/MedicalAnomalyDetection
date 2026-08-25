from __future__ import annotations

from fastapi import FastAPI

from app.db import Base, engine
from app.exceptions import AggregationError, NotFoundError, aggregation_error_handler, not_found_handler
from app.ml_interface import load_models
from app.routers import health, images, predictions, reports

app = FastAPI(title="Chest X-Ray Anomaly Detection API")

app.add_exception_handler(NotFoundError, not_found_handler)
app.add_exception_handler(AggregationError, aggregation_error_handler)

app.include_router(health.router)
app.include_router(images.router)
app.include_router(predictions.router)
app.include_router(reports.router)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    load_models()
