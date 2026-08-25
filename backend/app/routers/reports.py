from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.exceptions import NotFoundError
from app.models import Prediction, Report
from app.schemas import ReportResponse

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/{prediction_id}", response_model=ReportResponse, status_code=201)
async def create_report(prediction_id: int, db: Session = Depends(get_db)):
    pred = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not pred:
        raise NotFoundError(f"Prediction with id {prediction_id} not found")

    report = Report(prediction_id=prediction_id, pdf_path=None)
    db.add(report)
    db.commit()
    db.refresh(report)
    return report
