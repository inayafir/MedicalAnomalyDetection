from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.exceptions import NotFoundError
from app.models import Image, Patient, Prediction, Report
from app.reporting import generate_report_pdf
from app.schemas import ReportResponse
from app.storage import save_report

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/{prediction_id}", response_model=ReportResponse, status_code=201)
async def create_report(prediction_id: int, db: Session = Depends(get_db)):
    pred = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not pred:
        raise NotFoundError(f"Prediction with id {prediction_id} not found")

    image = db.query(Image).filter(Image.id == pred.image_id).first()
    if not image:
        raise NotFoundError(f"Image for prediction {prediction_id} not found")

    patient = (
        db.query(Patient).filter(Patient.id == image.patient_id).first()
        if image.patient_id
        else None
    )

    pdf_bytes = generate_report_pdf(prediction=pred, image=image, patient=patient)
    pdf_path = save_report(pdf_bytes, prediction_id)

    report = Report(prediction_id=prediction_id, pdf_path=pdf_path)
    db.add(report)
    db.commit()
    db.refresh(report)
    return report
