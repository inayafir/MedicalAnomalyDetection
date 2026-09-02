from __future__ import annotations

import io
import random

from PIL import Image

from app.config import settings
from app.models import CLASSIFIER_CLASSES, DETECTOR_CLASSES
from app.storage import save_heatmap

# Weighted distribution for the 5 classifier classes.
CLASS_WEIGHTS = {
    "Normal": 0.35,
    "Cardiomegaly": 0.20,
    "Pleural effusion": 0.20,
    "Lung Opacity": 0.15,
    "Pulmonary fibrosis": 0.10,
}


def mock_predict(image_path: str, original_width: int, original_height: int) -> dict:
    classes = list(CLASS_WEIGHTS.keys())
    weights = list(CLASS_WEIGHTS.values())
    chosen_class = random.choices(classes, weights=weights, k=1)[0]

    # Generate random bboxes using detector classes (not the classifier class)
    bboxes = []
    if chosen_class != "Normal":
        num_bboxes = random.randint(1, 3)
        for _ in range(num_bboxes):
            detector_class = random.choice(DETECTOR_CLASSES)
            bx1 = random.randint(0, original_width // 2)
            by1 = random.randint(0, original_height // 2)
            bx2 = random.randint(bx1 + 10, min(bx1 + 200, original_width))
            by2 = random.randint(by1 + 10, min(by1 + 200, original_height))
            bboxes.append({
                "class": detector_class,
                "x1": bx1,
                "y1": by1,
                "x2": bx2,
                "y2": by2,
                "confidence": round(random.uniform(0.60, 0.99), 2),
            })

    confidence = round(random.uniform(0.60, 0.99), 2)

    placeholder = Image.new("RGB", (64, 64), color=(
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    ))
    buf = io.BytesIO()
    placeholder.save(buf, format="PNG")
    heatmap_path = save_heatmap(buf.getvalue())

    return {
        "class": chosen_class,
        "confidence": confidence,
        "bboxes": bboxes,
        "heatmap_path": heatmap_path,
    }
