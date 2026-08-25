from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class PatientCreate(BaseModel):
    display_name: str | None = None


class PatientResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    display_name: str | None
    created_at: datetime


class ImageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_path: str
    original_filename: str
    content_type: str
    file_size_bytes: int
    uploaded_at: datetime
    patient_id: int | None


class BBox(BaseModel):
    class_: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    model_config = ConfigDict(populate_by_name=True)


class PredictionRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_id: int
    predicted_class: str
    confidence: float
    bboxes: list[dict]
    heatmap_path: str | None
    created_at: datetime


class ImageDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    original_filename: str
    uploaded_at: datetime
    latest_prediction: PredictionRecord | None = None


class ReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    prediction_id: int
    pdf_path: str | None
    generated_at: datetime


class HealthResponse(BaseModel):
    status: str
