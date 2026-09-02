from __future__ import annotations

import asyncio
import json
import os
from functools import partial

from fastapi import APIRouter, Depends, Query
from PIL import Image
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.exceptions import AggregationError, NotFoundError
from app.aggregation import build_prediction_record
from app.ml_interface import is_model_loaded, predict
from app.models import CLASSIFIER_CLASSES, Image as ImageModel, Prediction
from app.schemas import PaginatedResponse, PredictionListItem, PredictionRecord

router = APIRouter(prefix="/predictions", tags=["predictions"])

_VALID_CLASSIFIER = set(CLASSIFIER_CLASSES)


@router.post("/{image_id}", response_model=PredictionRecord, status_code=201)
async def create_prediction(image_id: int, db: Session = Depends(get_db)):
    if not is_model_loaded():
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="ML models not loaded — cannot run prediction")

    img = db.query(ImageModel).filter(ImageModel.id == image_id).first()
    if not img:
        raise NotFoundError(f"Image with id {image_id} not found")

    abs_path = os.path.join(settings.STORAGE_ROOT, img.file_path)
    with Image.open(abs_path) as pil_img:
        original_width, original_height = pil_img.size

    loop = asyncio.get_event_loop()
    raw_output = await loop.run_in_executor(
        None, partial(predict, img.file_path, original_width, original_height)
    )

    try:
        record_data = build_prediction_record(raw_output, image_id)
    except ValueError as e:
        raise AggregationError(str(e))

    prediction = Prediction(**record_data)
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return PredictionRecord(
        id=prediction.id,
        image_id=prediction.image_id,
        predicted_class=prediction.predicted_class,
        confidence=prediction.confidence,
        bboxes=json.loads(prediction.bboxes),
        heatmap_path=prediction.heatmap_path,
        created_at=prediction.created_at,
    )


@router.get("", response_model=PaginatedResponse[PredictionListItem])
async def list_predictions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    image_id: int | None = Query(None),
    predicted_class: str | None = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(Prediction)
    if image_id is not None:
        query = query.filter(Prediction.image_id == image_id)
    if predicted_class is not None:
        if predicted_class not in _VALID_CLASSIFIER:
            raise AggregationError(
                f"Invalid predicted_class '{predicted_class}'. "
                f"Valid classes: {sorted(_VALID_CLASSIFIER)}"
            )
        query = query.filter(Prediction.predicted_class == predicted_class)

    total = query.count()
    items = (
        query
        .order_by(Prediction.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return PaginatedResponse(
        items=[
            PredictionListItem(
                id=pred.id,
                image_id=pred.image_id,
                predicted_class=pred.predicted_class,
                confidence=pred.confidence,
                created_at=pred.created_at,
            )
            for pred in items
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{prediction_id}", response_model=PredictionRecord)
async def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    pred = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not pred:
        raise NotFoundError(f"Prediction with id {prediction_id} not found")

    return PredictionRecord(
        id=pred.id,
        image_id=pred.image_id,
        predicted_class=pred.predicted_class,
        confidence=pred.confidence,
        bboxes=json.loads(pred.bboxes),
        heatmap_path=pred.heatmap_path,
        created_at=pred.created_at,
    )
