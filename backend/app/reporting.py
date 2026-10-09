"""
PDF report generation for a completed prediction.

Built to look like a real radiology/imaging analysis report rather than an
academic project artifact — no student names, college name, or "prototype
demonstration" framing in the body. The safety disclaimer is present but
kept small, in the footer, as required for responsible AI-assisted output.

Every value in the report comes from real database records (Patient, Image,
Prediction) — nothing here is fabricated. If a value is genuinely missing,
the report says so explicitly rather than inventing one.
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
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.config import settings
from app.models import Image as ImageModel, Patient, Prediction

FOOTER_DISCLAIMER = (
    "AI-generated results are intended to assist clinical review and should "
    "be interpreted by a qualified healthcare professional."
)

# Palette — medical blue/teal, used sparingly and meaningfully (not decorative)
_BRAND = colors.HexColor("#0f6e8c")       # header / primary accent
_BRAND_LIGHT = colors.HexColor("#e6f3f7")  # light tint backgrounds
_INK = colors.HexColor("#1b2733")
_MUTED = colors.HexColor("#5c6b78")
_LINE = colors.HexColor("#dbe4ea")
_AMBER = colors.HexColor("#b96b00")


def _storage_path(relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    abs_path = (Path(settings.STORAGE_ROOT).resolve() / relative_path).resolve()
    return abs_path if abs_path.exists() else None


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(_MUTED)
    canvas.drawString(0.7 * inch, 0.45 * inch, FOOTER_DISCLAIMER)
    canvas.drawRightString(
        letter[0] - 0.7 * inch, 0.45 * inch, f"Page {doc.page}"
    )
    canvas.setStrokeColor(_LINE)
    canvas.line(0.7 * inch, 0.62 * inch, letter[0] - 0.7 * inch, 0.62 * inch)
    canvas.restoreState()


def generate_report_pdf(
    prediction: Prediction,
    image: ImageModel,
    patient: Patient | None,
) -> bytes:
    styles = getSampleStyleSheet()

    header_title = ParagraphStyle(
        "HeaderTitle", parent=styles["Title"], fontSize=17, textColor=_INK, spaceAfter=0
    )
    header_sub = ParagraphStyle(
        "HeaderSub", parent=styles["Normal"], fontSize=9, textColor=_MUTED
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=_BRAND,
        spaceBefore=16,
        spaceAfter=6,
    )
    prediction_label_style = ParagraphStyle(
        "PredLabel", parent=styles["Normal"], fontSize=9, textColor=colors.white
    )
    prediction_class_style = ParagraphStyle(
        "PredClass",
        parent=styles["Title"],
        fontSize=26,
        textColor=colors.white,
        spaceAfter=0,
        spaceBefore=2,
    )
    prediction_meta_style = ParagraphStyle(
        "PredMeta", parent=styles["Normal"], fontSize=9.5, textColor=colors.white
    )
    body_style = styles["Normal"]

    story = []

    # --- Header ---------------------------------------------------------
    story.append(Paragraph("Chest X-Ray Analysis Report", header_title))
    story.append(
        Paragraph(
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            header_sub,
        )
    )
    story.append(Spacer(1, 10))

    # --- Patient / study info table --------------------------------------
    patient_rows = [
        ["Patient ID", str(patient.id) if patient else "—"],
        ["Patient Name", patient.display_name if patient and patient.display_name else "—"],
        ["Study Image", image.original_filename],
        ["Study Date", image.uploaded_at.strftime("%Y-%m-%d %H:%M UTC")],
        ["Examination", "Chest X-Ray"],
    ]
    patient_table = Table(patient_rows, colWidths=[130, 340])
    patient_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (0, -1), _MUTED),
                ("TEXTCOLOR", (1, 0), (1, -1), _INK),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, _LINE),
            ]
        )
    )
    story.append(patient_table)

    # --- Primary prediction — the visual focal point of the report -------
    story.append(Spacer(1, 14))
    confidence_pct = prediction.confidence * 100
    pred_inner = Table(
        [
            [Paragraph("AI ANALYSIS &nbsp;&middot;&nbsp; PRIMARY CLASSIFICATION", prediction_label_style)],
            [Paragraph(prediction.predicted_class.upper(), prediction_class_style)],
            [Paragraph(f"Confidence: {confidence_pct:.1f}% &nbsp;&mdash;&nbsp; Model: ResNet-50", prediction_meta_style)],
        ],
        colWidths=[470],
    )
    pred_inner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _BRAND),
                ("LEFTPADDING", (0, 0), (-1, -1), 18),
                ("RIGHTPADDING", (0, 0), (-1, -1), 18),
                ("TOPPADDING", (0, 0), (0, 0), 12),
                ("BOTTOMPADDING", (0, 2), (0, 2), 12),
                ("TOPPADDING", (0, 1), (0, 1), 2),
                ("BOTTOMPADDING", (0, 1), (0, 1), 2),
            ]
        )
    )
    story.append(pred_inner)

    # --- Detected regions (YOLOv8m) — kept conceptually separate ---------
    story.append(Paragraph("Detected Regions (YOLOv8m)", section_style))
    story.append(
        Paragraph(
            "Localized abnormality candidates, independent of the primary "
            "classification above. Low-confidence entries are not confirmed findings.",
            ParagraphStyle("DetNote", parent=body_style, fontSize=8, textColor=_MUTED),
        )
    )
    story.append(Spacer(1, 4))
    bboxes = json.loads(prediction.bboxes or "[]")
    if bboxes:
        det_header = ["Region", "Confidence", "X1", "Y1", "X2", "Y2"]
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
        det_table = Table(det_rows, colWidths=[160, 80, 55, 55, 55, 55])
        det_table.setStyle(
            TableStyle(
                [
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("BACKGROUND", (0, 0), (-1, 0), _BRAND_LIGHT),
                    ("TEXTCOLOR", (0, 0), (-1, 0), _BRAND),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("GRID", (0, 0), (-1, -1), 0.25, _LINE),
                ]
            )
        )
        story.append(det_table)
    else:
        story.append(
            Paragraph(
                "No abnormality regions detected at the current confidence threshold.",
                body_style,
            )
        )

    # --- Images ------------------------------------------------------------
    story.append(Paragraph("Image Analysis", section_style))
    img_max_width = 2.4 * inch

    def _scaled(path: Path) -> RLImage:
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            w, h = im.size
        scale = img_max_width / w
        return RLImage(str(path), width=img_max_width, height=h * scale)

    img_cells = []
    captions = []
    original_path = _storage_path(image.file_path)
    if original_path is not None:
        img_cells.append(_scaled(original_path))
        captions.append("Original X-Ray")
    heatmap_path = _storage_path(prediction.heatmap_path)
    if heatmap_path is not None:
        img_cells.append(_scaled(heatmap_path))
        captions.append("Grad-CAM Explainability")

    if img_cells:
        img_table = Table([img_cells, captions], colWidths=[img_max_width + 10] * len(img_cells))
        img_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTSIZE", (0, 1), (-1, 1), 8),
                    ("TEXTCOLOR", (0, 1), (-1, 1), _MUTED),
                    ("TOPPADDING", (0, 1), (-1, 1), 4),
                ]
            )
        )
        story.append(KeepTogether(img_table))
    else:
        story.append(Paragraph("No images available for this study.", body_style))

    if heatmap_path is not None:
        story.append(Spacer(1, 4))
        story.append(
            Paragraph(
                "Grad-CAM highlights the image regions that most influenced the "
                "ResNet-50 classification decision.",
                ParagraphStyle("GcNote", parent=body_style, fontSize=8, textColor=_MUTED),
            )
        )

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        topMargin=0.55 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        title="Chest X-Ray Analysis Report",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()