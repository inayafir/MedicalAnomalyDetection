"""
Renders bounding boxes (already computed by the real YOLOv8m backend) onto a
copy of the original image for display. This is drawing only — it never
computes, adjusts, or invents any detection itself.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

_BOX_COLOR = (255, 64, 64)
_TEXT_BG = (255, 64, 64)
_TEXT_COLOR = (255, 255, 255)


def draw_bboxes(image: Image.Image, bboxes: list[dict]) -> Image.Image:
    """Returns a NEW image with detection boxes + labels drawn on top."""
    annotated = image.convert("RGB").copy()
    if not bboxes:
        return annotated

    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()

    for box in bboxes:
        cls = box.get("class") or box.get("class_", "?")
        conf = box.get("confidence", 0.0)
        x1, y1, x2, y2 = (
            box.get("x1", 0),
            box.get("y1", 0),
            box.get("x2", 0),
            box.get("y2", 0),
        )
        draw.rectangle([x1, y1, x2, y2], outline=_BOX_COLOR, width=3)

        label = f"{cls} {conf * 100:.0f}%"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, y1 - text_h - 6)
        draw.rectangle(
            [x1, label_y, x1 + (text_bbox[2] - text_bbox[0]) + 8, label_y + text_h + 6],
            fill=_TEXT_BG,
        )
        draw.text((x1 + 4, label_y + 2), label, fill=_TEXT_COLOR, font=font)

    return annotated
