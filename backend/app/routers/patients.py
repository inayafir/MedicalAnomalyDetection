from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Patient
from app.schemas import PaginatedResponse, PatientCreate, PatientResponse

router = APIRouter(prefix="/patients", tags=["patients"])


@router.post("", response_model=PatientResponse, status_code=201)
async def create_patient(body: PatientCreate, db: Session = Depends(get_db)):
    patient = Patient(display_name=body.display_name)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@router.get("", response_model=PaginatedResponse[PatientResponse])
async def list_patients(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    query = db.query(Patient)
    total = query.count()
    items = (
        query
        .order_by(Patient.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return PaginatedResponse(
        items=[
            PatientResponse(
                id=p.id,
                display_name=p.display_name,
                created_at=p.created_at,
            )
            for p in items
        ],
        total=total,
        limit=limit,
        offset=offset,
    )
