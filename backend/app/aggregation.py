from __future__ import annotations

import json

from app.models import FindingClass


def build_prediction_record(raw_ml_output: dict, image_id: int) -> dict:
    """Validate raw ML output and return a dict ready for DB persistence."""
    if not isinstance(raw_ml_output, dict):
        raise ValueError("ML output must be a dict")

    class_name = raw_ml_output.get("class")
    if not class_name or class_name not in [c.value for c in FindingClass]:
        raise ValueError(f"Invalid class: {class_name}")

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

    heatmap_path = raw_ml_output.get("heatmap_path")

    return {
        "image_id": image_id,
        "predicted_class": class_name,
        "confidence": float(confidence),
        "bboxes": json.dumps(bboxes),
        "heatmap_path": heatmap_path,
    }
