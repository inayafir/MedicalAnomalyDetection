from __future__ import annotations

import json
import os

from fastapi import APIRouter, Depends
from PIL import Image
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.exceptions import AggregationError, NotFoundError
from app.aggregation import build_prediction_record
from app.ml_interface import predict
from app.models import Image as ImageModel, Prediction
from app.schemas import PredictionRecord

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/{image_id}", response_model=PredictionRecord, status_code=201)
async def create_prediction(image_id: int, db: Session = Depends(get_db)):
    img = db.query(ImageModel).filter(ImageModel.id == image_id).first()
    if not img:
        raise NotFoundError(f"Image with id {image_id} not found")

    abs_path = os.path.join(settings.STORAGE_ROOT, img.file_path)
    with Image.open(abs_path) as pil_img:
        original_width, original_height = pil_img.size

    raw_output = predict(img.file_path, original_width, original_height)

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
