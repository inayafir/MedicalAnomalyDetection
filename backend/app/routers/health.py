from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.ml_interface import is_model_loaded
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    db_ok = False
    try:
        db.execute(__import__("sqlalchemy", fromlist=["text"]).text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        model_loaded=is_model_loaded(),
        db_ok=db_ok,
    )


@router.get("/health/live")
async def liveness():
    return {"status": "ok"}


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(db: Session = Depends(get_db)):
    db_ok = False
    try:
        db.execute(__import__("sqlalchemy", fromlist=["text"]).text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    models_ok = is_model_loaded()
    return HealthResponse(
        status="ok" if (db_ok and models_ok) else "degraded",
        model_loaded=models_ok,
        db_ok=db_ok,
    )
