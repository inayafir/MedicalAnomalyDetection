from __future__ import annotations

import json

from app.models import CLASSIFIER_CLASSES


_VALID_CLASSIFIER = set(CLASSIFIER_CLASSES)


def build_prediction_record(raw_ml_output: dict, image_id: int) -> dict:
    """Validate raw ML output and return a dict ready for DB persistence.

    Top-level `class` is validated against CLASSIFIER_CLASSES (5).
    Bbox `class` fields are validated as non-empty strings (14 detector classes
    are not enum-constrained here — they come from the YOLO model's own labels).
    """
    if not isinstance(raw_ml_output, dict):
        raise ValueError("ML output must be a dict")

    class_name = raw_ml_output.get("class")
    if not class_name or class_name not in _VALID_CLASSIFIER:
        raise ValueError(
            f"Invalid classifier class: {class_name!r}. "
            f"Expected one of: {sorted(_VALID_CLASSIFIER)}"
        )

    confidence = raw_ml_output.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
        raise ValueError(f"Confidence must be a float in [0,1], got {confidence}")

    bboxes = raw_ml_output.get("bboxes")
    if not isinstance(bboxes, list):
        raise ValueError("bboxes must be a list")

    for bbox in bboxes:
        for key in ("class", "x1", "y1", "x2", "y2", "confidence"):
            if key not in bbox:
                raise ValueError(f"bbox missing required key: {key}")
        bbox_class = bbox.get("class")
        if not isinstance(bbox_class, str) or not bbox_class:
            raise ValueError(f"bbox class must be a non-empty string, got {bbox_class!r}")

    heatmap_path = raw_ml_output.get("heatmap_path")

    return {
        "image_id": image_id,
        "predicted_class": class_name,
        "confidence": float(confidence),
        "bboxes": json.dumps(bboxes),
        "heatmap_path": heatmap_path,
    }
