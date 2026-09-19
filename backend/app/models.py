from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.db import Base


# ---------------------------------------------------------------------------
# Two separate label sets — ResNet-50 classifier (15) and YOLOv8 detector (14).
#
# The 14 detector classes are the 14 VinBigData abnormality findings.
# The 15 classifier classes are those same 14 findings plus "Normal" (i.e.
# "no finding detected" as a whole-image label).
#
# The two models serve different purposes:
#   - ResNet classifies the whole image → top-level `class` field.
#   - YOLO detects abnormality regions → per-bbox `class` field.
#
# They are kept intentionally separate. A classifier can say "Normal" while
# YOLO still finds a small region — or vice versa. This is valid model
# disagreement, not an error.
# ---------------------------------------------------------------------------
CLASSIFIER_CLASSES: list[str] = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Normal",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]

DETECTOR_CLASSES: list[str] = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]


class ClassifierClass(str, enum.Enum):
    """Whole-image label from ResNet-50 (15 classes: 14 findings + Normal)."""

    AORTIC_ENLARGEMENT = "Aortic enlargement"
    ATELECTASIS = "Atelectasis"
    CALCIFICATION = "Calcification"
    CARDIOMEGALY = "Cardiomegaly"
    CONSOLIDATION = "Consolidation"
    ILD = "ILD"
    INFILTRATION = "Infiltration"
    LUNG_OPACITY = "Lung Opacity"
    NODULE_MASS = "Nodule/Mass"
    NORMAL = "Normal"
    OTHER_LESION = "Other lesion"
    PLEURAL_EFFUSION = "Pleural effusion"
    PLEURAL_THICKENING = "Pleural thickening"
    PNEUMOTHORAX = "Pneumothorax"
    PULMONARY_FIBROSIS = "Pulmonary fibrosis"


class DetectorClass(str, enum.Enum):
    """Per-region label from YOLOv8 detector (14 abnormality classes only)."""

    AORTIC_ENLARGEMENT = "Aortic enlargement"
    ATELECTASIS = "Atelectasis"
    CALCIFICATION = "Calcification"
    CARDIOMEGALY = "Cardiomegaly"
    CONSOLIDATION = "Consolidation"
    ILD = "ILD"
    INFILTRATION = "Infiltration"
    LUNG_OPACITY = "Lung Opacity"
    NODULE_MASS = "Nodule/Mass"
    OTHER_LESION = "Other lesion"
    PLEURAL_EFFUSION = "Pleural effusion"
    PLEURAL_THICKENING = "Pleural thickening"
    PNEUMOTHORAX = "Pneumothorax"
    PULMONARY_FIBROSIS = "Pulmonary fibrosis"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    display_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    images = relationship("Image", back_populates="patient")


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True, index=True)
    file_path = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    patient = relationship("Patient", back_populates="images")
    predictions = relationship(
        "Prediction", back_populates="image", cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(
        Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    predicted_class = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    bboxes = Column(Text, nullable=False, default="[]")
    heatmap_path = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    image = relationship("Image", back_populates="predictions")
    reports = relationship(
        "Report", back_populates="prediction", cascade="all, delete-orphan"
    )


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(
        Integer, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False
    )
    pdf_path = Column(String, nullable=True)
    generated_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    prediction = relationship("Prediction", back_populates="reports")
