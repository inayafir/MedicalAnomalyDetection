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
