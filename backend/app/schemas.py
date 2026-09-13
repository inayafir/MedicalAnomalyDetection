from __future__ import annotations

from datetime import datetime
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------

class PatientCreate(BaseModel):
    display_name: str | None = None


class PatientResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    display_name: str | None
    created_at: datetime


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------

class BBox(BaseModel):
    class_: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    model_config = ConfigDict(populate_by_name=True, alias="class")

    @model_validator(mode="before")
    @classmethod
    def normalize_class_key(cls, data):
        if isinstance(data, dict) and "class" in data and "class_" not in data:
            data["class_"] = data.pop("class")
        return data


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class PredictionRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_id: int
    predicted_class: str
    confidence: float
    bboxes: list[BBox]
    heatmap_path: str | None
    created_at: datetime


class PredictionListItem(BaseModel):
    """Compact prediction for list views."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_id: int
    predicted_class: str
    confidence: float
    created_at: datetime


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

class ImageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_path: str
    original_filename: str
    content_type: str
    file_size_bytes: int
    uploaded_at: datetime
    patient_id: int | None


class ImageListItem(BaseModel):
    """Compact image for list views (no latest_prediction embedded)."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_path: str
    original_filename: str
    content_type: str
    file_size_bytes: int
    uploaded_at: datetime
    patient_id: int | None


class ImageDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_path: str
    original_filename: str
    content_type: str
    file_size_bytes: int
    uploaded_at: datetime
    patient_id: int | None
    latest_prediction: PredictionRecord | None = None


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

class ReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    prediction_id: int
    pdf_path: str | None
    generated_at: datetime


# ---------------------------------------------------------------------------
# Pagination envelope
# ---------------------------------------------------------------------------

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    db_ok: bool
