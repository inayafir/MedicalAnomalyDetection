from __future__ import annotations

import io
import random

from PIL import Image

from app.config import settings
from app.models import FindingClass
from app.storage import save_heatmap

# Weighted distribution matching realistic chest X-ray class imbalance.
# Normal is most common; rare findings get lower weights.
CLASS_WEIGHTS = {
    FindingClass.NORMAL: 0.30,
    FindingClass.CARDIOMEGALY: 0.10,
    FindingClass.PLEURAL_EFFUSION: 0.10,
    FindingClass.LUNG_OPACITY: 0.10,
    FindingClass.PULMONARY_FIBROSIS: 0.06,
    FindingClass.AORTIC_ENLARGEMENT: 0.05,
    FindingClass.ATELECTASIS: 0.05,
    FindingClass.CALCIFICATION: 0.04,
    FindingClass.CONSOLIDATION: 0.04,
    FindingClass.ILD: 0.04,
    FindingClass.INFILTRATION: 0.04,
    FindingClass.NODULE_MASS: 0.04,
    FindingClass.OTHER_LESION: 0.03,
    FindingClass.PLEURAL_THICKENING: 0.03,
    FindingClass.PNEUMOTHORAX: 0.02,
}

# Only disease classes produce bounding boxes (not Normal)
BBOX_CLASSES = {
    FindingClass.AORTIC_ENLARGEMENT,
    FindingClass.ATELECTASIS,
    FindingClass.CALCIFICATION,
    FindingClass.CARDIOMEGALY,
    FindingClass.CONSOLIDATION,
    FindingClass.ILD,
    FindingClass.INFILTRATION,
    FindingClass.LUNG_OPACITY,
    FindingClass.NODULE_MASS,
    FindingClass.OTHER_LESION,
    FindingClass.PLEURAL_EFFUSION,
    FindingClass.PLEURAL_THICKENING,
    FindingClass.PNEUMOTHORAX,
    FindingClass.PULMONARY_FIBROSIS,
}


def mock_predict(image_path: str, original_width: int, original_height: int) -> dict:
    classes = list(CLASS_WEIGHTS.keys())
    weights = list(CLASS_WEIGHTS.values())
    chosen_class = random.choices(classes, weights=weights, k=1)[0]

    bboxes = []
    if chosen_class in BBOX_CLASSES:
        num_bboxes = random.randint(1, 3)
        for _ in range(num_bboxes):
            bx1 = random.randint(0, original_width // 2)
            by1 = random.randint(0, original_height // 2)
            bx2 = random.randint(bx1 + 10, min(bx1 + 200, original_width))
            by2 = random.randint(by1 + 10, min(by1 + 200, original_height))
            bboxes.append(
                {
                    "class": chosen_class.value,
                    "x1": bx1,
                    "y1": by1,
                    "x2": bx2,
                    "y2": by2,
                    "confidence": round(random.uniform(0.60, 0.99), 2),
                }
            )

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
        "class": chosen_class.value,
        "confidence": confidence,
        "bboxes": bboxes,
        "heatmap_path": heatmap_path,
    }
