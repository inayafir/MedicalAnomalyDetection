"""
PDF report generation for a completed prediction.

Builds a real PDF from actual database records (Patient, Image, Prediction) —
never fabricates classification results, confidence values, or detections.
If a piece of data is genuinely missing (e.g. no patient linked), the report
says so explicitly rather than inventing a value.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.config import settings
from app.models import Image as ImageModel, Patient, Prediction

DISCLAIMER = (
    "This system is an academic/decision-support prototype and is not a "
    "medical diagnosis. Results should be reviewed by a qualified medical "
    "professional before any clinical decision is made."
)


def _storage_path(relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    abs_path = (Path(settings.STORAGE_ROOT).resolve() / relative_path).resolve()
    return abs_path if abs_path.exists() else None


def generate_report_pdf(
    prediction: Prediction,
    image: ImageModel,
    patient: Patient | None,
) -> bytes:
    """Build the PDF and return its raw bytes. Raises on missing required data —
    never silently substitutes placeholder/fake content for a real result."""
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=18, spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], fontSize=10, textColor=colors.grey
    )
    section_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#555555"),
        borderColor=colors.HexColor("#999999"),
        borderWidth=0.5,
        borderPadding=6,
    )

    story = []

    # --- Header ---------------------------------------------------------
    story.append(Paragraph("Chest X-Ray Analysis Report", title_style))
    story.append(
        Paragraph(
            "AI-assisted classification and detection — ResNet-50 + YOLOv8m + Grad-CAM",
            subtitle_style,
        )
    )
    story.append(
        Paragraph(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 10))
    story.append(Paragraph(DISCLAIMER, disclaimer_style))
    story.append(Spacer(1, 12))

    # --- Patient information --------------------------------------------
    story.append(Paragraph("Patient Information", section_style))
    patient_rows = [
        ["Patient ID", str(patient.id) if patient else "Not provided"],
        ["Name", patient.display_name if patient and patient.display_name else "Not provided"],
        ["Study Image", image.original_filename],
        ["Uploaded", image.uploaded_at.strftime("%Y-%m-%d %H:%M UTC")],
    ]
    patient_table = Table(patient_rows, colWidths=[150, 320])
    patient_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#444444")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
            ]
        )
    )
    story.append(patient_table)

    # --- Classification result -------------------------------------------
    story.append(Paragraph("Classification Result (ResNet-50)", section_style))
    class_rows = [
        ["Predicted Class", prediction.predicted_class],
        ["Confidence", f"{prediction.confidence * 100:.1f}%"],
    ]
    class_table = Table(class_rows, colWidths=[150, 320])
    class_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
            ]
        )
    )
    story.append(class_table)

    # --- Detections (YOLOv8) ---------------------------------------------
    story.append(Paragraph("Detected Abnormalities (YOLOv8m)", section_style))
    bboxes = json.loads(prediction.bboxes or "[]")
    if bboxes:
        det_header = ["Class", "Confidence", "X1", "Y1", "X2", "Y2"]
        det_rows = [det_header] + [
            [
                b.get("class") or b.get("class_", ""),
                f"{float(b.get('confidence', 0)) * 100:.1f}%",
                str(b.get("x1", "")),
                str(b.get("y1", "")),
                str(b.get("x2", "")),
                str(b.get("y2", "")),
            ]
            for b in bboxes
        ]
        det_table = Table(det_rows, colWidths=[150, 80, 55, 55, 55, 55])
        det_table.setStyle(
            TableStyle(
                [
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
                ]
            )
        )
        story.append(det_table)
    else:
        story.append(
            Paragraph(
                "No abnormalities detected by the YOLOv8m model at the current "
                "confidence threshold.",
                styles["Normal"],
            )
        )

    # --- Images ------------------------------------------------------------
    story.append(Paragraph("Imaging", section_style))
    img_max_width = 3.2 * inch

    original_path = _storage_path(image.file_path)
    if original_path is not None:
        story.append(Paragraph("Original X-Ray", styles["Heading4"]))
        story.append(_scaled_image(original_path, img_max_width))
        story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("Original X-ray image file not found on disk.", styles["Normal"]))

    heatmap_path = _storage_path(prediction.heatmap_path)
    if heatmap_path is not None:
        story.append(Paragraph("Grad-CAM Explainability Heatmap", styles["Heading4"]))
        story.append(_scaled_image(heatmap_path, img_max_width))
    else:
        story.append(
            Paragraph(
                "Grad-CAM heatmap was not generated for this prediction.", styles["Normal"]
            )
        )

    story.append(Spacer(1, 16))
    story.append(Paragraph(DISCLAIMER, disclaimer_style))

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        title="Chest X-Ray Analysis Report",
    )
    doc.build(story)
    return buf.getvalue()


def _scaled_image(path: Path, max_width: float) -> RLImage:
    """Load an image sized to max_width, preserving aspect ratio."""
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    scale = max_width / w
    return RLImage(str(path), width=max_width, height=h * scale)
