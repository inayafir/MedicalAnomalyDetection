from __future__ import annotations

import io
import json
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from PIL import Image
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.exceptions import NotFoundError
from app.models import Image as ImageModel, Patient, Prediction
from app.schemas import ImageDetail, ImageListItem, ImageResponse, PaginatedResponse, PredictionRecord
from app.storage import delete_file, save_image

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/upload", response_model=ImageResponse, status_code=201)
async def upload_image(
    file: UploadFile,
    patient_id: int | None = None,
    db: Session = Depends(get_db),
):
    if file.content_type not in settings.allowed_content_types_list:
        raise HTTPException(
            status_code=415,
            detail=f"Content type '{file.content_type}' not allowed. Use: {settings.ALLOWED_CONTENT_TYPES}",
        )

    content = await file.read()
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="File is not a valid image.",
        )

    if patient_id is not None:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise NotFoundError(f"Patient with id {patient_id} not found")

    file.file.seek(0)
    rel_path = save_image(file)

    img_record = ImageModel(
        patient_id=patient_id,
        file_path=rel_path,
        original_filename=file.filename or "unknown",
        content_type=file.content_type,
        file_size_bytes=len(content),
    )
    db.add(img_record)
    db.commit()
    db.refresh(img_record)
    return img_record


@router.get("", response_model=PaginatedResponse[ImageListItem])
async def list_images(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    patient_id: int | None = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(ImageModel)
    if patient_id is not None:
        query = query.filter(ImageModel.patient_id == patient_id)

    total = query.count()
    items = (
        query
        .order_by(ImageModel.uploaded_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return PaginatedResponse(
        items=[
            ImageListItem(
                id=img.id,
                file_path=img.file_path,
                original_filename=img.original_filename,
                content_type=img.content_type,
                file_size_bytes=img.file_size_bytes,
                uploaded_at=img.uploaded_at,
                patient_id=img.patient_id,
            )
            for img in items
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{image_id}", response_model=ImageDetail)
async def get_image(image_id: int, db: Session = Depends(get_db)):
    img = db.query(ImageModel).filter(ImageModel.id == image_id).first()
    if not img:
        raise NotFoundError(f"Image with id {image_id} not found")

    latest_pred = (
        db.query(Prediction)
        .filter(Prediction.image_id == image_id)
        .order_by(Prediction.created_at.desc())
        .first()
    )

    latest_pred_schema = None
    if latest_pred:
        latest_pred_schema = PredictionRecord(
            id=latest_pred.id,
            image_id=latest_pred.image_id,
            predicted_class=latest_pred.predicted_class,
            confidence=latest_pred.confidence,
            bboxes=json.loads(latest_pred.bboxes),
            heatmap_path=latest_pred.heatmap_path,
            created_at=latest_pred.created_at,
        )

    return ImageDetail(
        id=img.id,
        file_path=img.file_path,
        original_filename=img.original_filename,
        content_type=img.content_type,
        file_size_bytes=img.file_size_bytes,
        uploaded_at=img.uploaded_at,
        patient_id=img.patient_id,
        latest_prediction=latest_pred_schema,
    )


@router.delete("/{image_id}", status_code=204)
async def delete_image(image_id: int, db: Session = Depends(get_db)):
    img = db.query(ImageModel).filter(ImageModel.id == image_id).first()
    if not img:
        raise NotFoundError(f"Image with id {image_id} not found")

    # Delete underlying files (image + any heatmaps from predictions)
    delete_file(img.file_path)
    for pred in img.predictions:
        if pred.heatmap_path:
            delete_file(pred.heatmap_path)

    db.delete(img)
    db.commit()
    return None
