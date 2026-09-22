"""
Gradio UI for the Medical X-Ray Anomaly Detection project.

Guided workflow:
Welcome -> Patient/X-ray Input -> Analysis -> Results -> Report

This file contains NO model-loading or inference logic.
All classification, detection, Grad-CAM, and PDF report generation
come from the existing FastAPI backend.

Run with:
    python gradio_app/app.py

Backend:
    uvicorn app.main:app
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import tempfile

import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from services import api_client  # noqa: E402
from services.overlay import draw_bboxes  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gradio_app")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FOOTER_DISCLAIMER = (
    "AI-generated results are intended to assist clinical review and should be "
    "interpreted by a qualified healthcare professional."
)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_FILE_MB = 10

SCREEN_NAMES = [
    "welcome",
    "input",
    "analysis",
    "results",
    "report",
]


# ---------------------------------------------------------------------------
# Visual design system
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
:root {
    --navy: #0B3D91;
    --navy-dark: #062A66;
    --teal: #0EA5A0;
    --cyan: #22D3EE;
    --coral: #FB7185;
    --bg: #EEF4FF;
    --card: #FFFFFF;
    --ink: #0F172A;
    --muted: #64748B;
    --success: #16A34A;
    --success-light: #ECFDF3;
    --warning: #F59E0B;
    --warning-light: #FFFBEB;
    --line: #E2E8F0;
}

/* ------------------------------------------------------------------ */
/* Force light mode — stops Gradio's dark theme from turning the      */
/* Patient Info / Upload blocks black on dark-mode browsers/OS        */
/* ------------------------------------------------------------------ */

:root, .gradio-container {
    color-scheme: light !important;
}

/* ------------------------------------------------------------------ */
/* Override Gradio's own theme variables — this is what actually      */
/* controls the blue "block label" chips (Patient Name / Patient ID / */
/* Upload X-Ray badges) and the faint upload dropzone text.           */
/* ------------------------------------------------------------------ */

.gradio-container {
    --block-label-background-fill: transparent !important;
    --block-label-text-color: var(--muted) !important;
    --block-label-border-width: 0px !important;
    --block-label-shadow: none !important;
    --block-label-margin: 0 !important;

    --body-text-color: var(--ink) !important;
    --body-text-color-subdued: var(--muted) !important;

    --input-background-fill: var(--card) !important;
    --input-background-fill-focus: var(--card) !important;
    --input-border-color: var(--line) !important;
    --input-border-color-focus: var(--teal) !important;
    --input-placeholder-color: var(--muted) !important;

    --block-background-fill: var(--card) !important;
    --block-border-color: var(--line) !important;
    --border-color-primary: var(--line) !important;

    --neutral-50: var(--card) !important;
    --neutral-100: var(--card) !important;
}

/* Label chip text (was rendering as a solid blue badge) */
.gradio-container label span,
.gradio-container .block > label,
.gradio-container [data-testid="block-info"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    border: none !important;
}

/* Upload dropzone icon + "Drop File Here / - or - / Click to Upload" text */
.gradio-container .upload-container,
.gradio-container .upload-container *,
.gradio-container [data-testid="file"] .wrap,
.gradio-container [data-testid="file"] .wrap * {
    color: var(--ink) !important;
    opacity: 1 !important;
}

.gradio-container .upload-container svg,
.gradio-container [data-testid="file"] svg {
    stroke: var(--muted) !important;
    fill: var(--muted) !important;
}

.gradio-container {
    background: linear-gradient(180deg, #EFF6FF 0%, #F0FDFA 55%, #F8FAFC 100%) !important;
}

/* Raw Gradio blocks (rows/columns wrapping our inputs) */
.gradio-container .block,
.gradio-container .form,
.gr-box,
.gr-panel,
.gradio-container .gr-group,
.gradio-container fieldset {
    background: var(--card) !important;
    color: var(--ink) !important;
    border-color: var(--line) !important;
}

/* Textboxes (Patient Name / Patient ID) */
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container textarea {
    background: var(--card) !important;
    color: var(--ink) !important;
    border-color: var(--line) !important;
}

.gradio-container input[type="text"]::placeholder,
.gradio-container textarea::placeholder {
    color: var(--muted) !important;
}

/* Labels above inputs */
.gradio-container label span,
.gradio-container .block > label {
    color: var(--ink) !important;
}

/* File upload drop zone */
.gradio-container [data-testid="file"],
.gradio-container .upload-container,
.gradio-container .upload-box,
.gradio-container .wrap.default {
    background: var(--card) !important;
    color: var(--ink) !important;
    border-color: var(--line) !important;
}

/* Image preview component frame */
.gradio-container .image-container,
.gradio-container .image-frame {
    background: var(--card) !important;
}

footer {
    visibility: hidden;
}

.screen-title {
    color: var(--navy);
    font-size: 1.7em;
    font-weight: 800;
    margin-bottom: 2px;
    letter-spacing: -0.01em;
}

.screen-subtitle {
    color: var(--muted);
    font-size: 0.95em;
    margin-bottom: 4px;
}

/* ------------------------------------------------------------------ */
/* Welcome / hero                                                      */
/* ------------------------------------------------------------------ */

.welcome-hero {
    background: linear-gradient(
        120deg,
        var(--navy) 0%,
        var(--navy-dark) 35%,
        var(--teal) 100%
    );
    color: white;
    border-radius: 22px;
    padding: 56px 40px 44px 40px;
    text-align: center;
    margin: 10px 0 22px 0;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 40px -12px rgba(11, 61, 145, 0.45);
}

.welcome-hero::before {
    content: "";
    position: absolute;
    top: -60px;
    right: -60px;
    width: 220px;
    height: 220px;
    background: radial-gradient(circle, rgba(34, 211, 238, 0.35) 0%, transparent 70%);
    border-radius: 50%;
}

.welcome-hero::after {
    content: "";
    position: absolute;
    bottom: -80px;
    left: -40px;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(251, 113, 133, 0.20) 0%, transparent 70%);
    border-radius: 50%;
}

.welcome-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.28);
    color: #E0F2FE;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 0.78em;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 18px;
    position: relative;
    z-index: 1;
}

.welcome-hero h1 {
    margin: 0 0 12px 0;
    font-size: 2.3em;
    font-weight: 800;
    position: relative;
    z-index: 1;
    letter-spacing: -0.02em;
}

.welcome-hero p {
    margin: 0 auto;
    opacity: 0.92;
    font-size: 1.05em;
    max-width: 560px;
    position: relative;
    z-index: 1;
    line-height: 1.5;
}

.feature-strip {
    display: flex;
    justify-content: center;
    gap: 14px;
    flex-wrap: wrap;
    margin-top: 26px;
    position: relative;
    z-index: 1;
}

.feature-pill {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.22);
    backdrop-filter: blur(4px);
    border-radius: 14px;
    padding: 10px 18px;
    font-size: 0.88em;
    font-weight: 600;
    color: #F0F9FF;
    display: flex;
    align-items: center;
    gap: 8px;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 18px;
}

@media (max-width: 900px) {
    .feature-grid { grid-template-columns: 1fr; }
}

.feature-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 22px 20px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    transition: transform 0.15s ease;
}

.feature-card .icon-circle {
    width: 52px;
    height: 52px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6em;
    margin: 0 auto 12px auto;
}

.feature-card .icon-circle.blue {
    background: linear-gradient(135deg, #DBEAFE, #BFDBFE);
}

.feature-card .icon-circle.teal {
    background: linear-gradient(135deg, #CCFBF1, #99F6E4);
}

.feature-card .icon-circle.coral {
    background: linear-gradient(135deg, #FFE4E6, #FECDD3);
}

.feature-card h4 {
    margin: 0 0 6px 0;
    color: var(--navy);
    font-size: 1em;
    font-weight: 700;
}

.feature-card p {
    margin: 0;
    color: var(--muted);
    font-size: 0.85em;
    line-height: 1.4;
}

.disclaimer-banner {
    background: var(--warning-light);
    border: 1px solid #FDE68A;
    color: #92400E;
    border-radius: 12px;
    padding: 12px 18px;
    font-size: 0.85em;
    text-align: center;
    margin-bottom: 6px;
}

/* ------------------------------------------------------------------ */
/* Cards                                                                */
/* ------------------------------------------------------------------ */

.card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    margin-bottom: 14px;
}

.card h3 {
    margin: 0 0 8px 0;
    color: var(--navy);
    font-size: 1.05em;
    font-weight: 700;
}

.card-note {
    color: var(--muted);
    font-size: 0.85em;
    margin: 0 0 8px 0;
}

/* Upload */

.upload-status-ok {
    background: var(--success-light);
    color: var(--success);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.9em;
    font-weight: 600;
}

.upload-status-bad {
    background: var(--warning-light);
    color: var(--warning);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.9em;
    font-weight: 600;
}

/* Analysis checklist */

.checklist-item {
    padding: 4px 0;
    font-size: 0.98em;
    color: var(--ink);
}

.checklist-done {
    color: var(--success);
}

.checklist-pending {
    color: var(--muted);
}

/* Prediction */

.prediction-card {
    background: linear-gradient(
        120deg,
        var(--navy) 0%,
        var(--navy-dark) 40%,
        var(--teal) 100%
    );
    color: white;
    border-radius: 16px;
    padding: 24px 28px;
    margin: 6px 0 16px 0;
    box-shadow: 0 14px 30px -10px rgba(11, 61, 145, 0.4);
}

.prediction-card .kicker {
    font-size: 0.78em;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.85;
    margin-bottom: 6px;
}

.prediction-card .cls {
    font-size: 2.1em;
    font-weight: 800;
}

.prediction-card .conf {
    margin-top: 8px;
    font-size: 1.05em;
    color: var(--cyan);
    font-weight: 700;
}

.prediction-card .model-tag {
    margin-top: 2px;
    font-size: 0.82em;
    opacity: 0.8;
}

/* Badges */

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.8em;
    font-weight: 600;
}

.badge-success {
    background: var(--success-light);
    color: var(--success);
}

.badge-warning {
    background: var(--warning-light);
    color: var(--warning);
}

/* Footer */

.disclaimer-footer {
    color: var(--muted);
    font-size: 0.78em;
    text-align: center;
    padding: 12px 0 4px 0;
    border-top: 1px solid var(--line);
    margin-top: 20px;
}

/* Tables */

.detection-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92em;
}

.detection-table th {
    text-align: left;
    color: var(--muted);
    padding: 8px;
    border-bottom: 1px solid var(--line);
}

.detection-table td {
    padding: 8px;
    border-bottom: 1px solid var(--line);
    color: var(--ink);
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_error(exc: Exception) -> str:
    logger.exception("Operation failed")

    if isinstance(exc, api_client.BackendError):
        return f"⚠️ {exc}"

    return (
        "⚠️ Something went wrong. "
        "Please try again or check the backend logs."
    )


def _validate_upload(file_path: str | None) -> str | None:
    if not file_path:
        return "Please upload an X-ray image."

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return (
            f"Unsupported format '{ext}'. "
            "Please use PNG or JPEG."
        )

    size_mb = os.path.getsize(file_path) / (1024 * 1024)

    if size_mb > MAX_FILE_MB:
        return (
            f"File is {size_mb:.1f} MB — "
            f"please upload an image under {MAX_FILE_MB} MB."
        )

    return None


def _checklist_html(done_steps: int) -> str:
    steps = [
        "Image uploaded",
        "Image preprocessing",
        "ResNet-50 classification",
        "YOLOv8 detection",
        "Grad-CAM generation",
    ]

    lines = []

    for i, label in enumerate(steps):
        if i < done_steps:
            lines.append(
                f'<div class="checklist-item checklist-done">'
                f"✓ {label}"
                f"</div>"
            )

        elif i == done_steps:
            lines.append(
                f'<div class="checklist-item">'
                f"● {label}"
                f"</div>"
            )

        else:
            lines.append(
                f'<div class="checklist-item checklist-pending">'
                f"○ {label}"
                f"</div>"
            )

    return (
        '<div class="card">'
        "<h3>Analyzing X-Ray</h3>"
        + "".join(lines)
        + "</div>"
    )


def _prediction_card_html(
    predicted_class: str,
    confidence: float,
) -> str:

    return f"""
    <div class="prediction-card">
        <div class="kicker">Primary Classification</div>
        <div class="cls">{predicted_class}</div>
        <div class="conf">
            {confidence * 100:.1f}% confidence
        </div>
        <div class="model-tag">ResNet-50</div>
    </div>
    """


def _detections_table_html(bboxes: list[dict]) -> str:

    if not bboxes:
        return (
            '<div class="card">'
            "<h3>Detected Abnormalities (YOLOv8m)</h3>"
            '<span class="badge badge-success">'
            "No regions detected"
            "</span>"
            "</div>"
        )

    rows = ""

    for b in bboxes:

        class_name = (
            b.get("class")
            or b.get("class_")
            or "Unknown"
        )

        confidence = float(
            b.get("confidence", 0)
        )

        x1 = b.get("x1", "?")
        y1 = b.get("y1", "?")
        x2 = b.get("x2", "?")
        y2 = b.get("y2", "?")

        rows += f"""
        <tr>
            <td>{class_name}</td>
            <td>{confidence * 100:.1f}%</td>
            <td>
                ({x1}, {y1}) - ({x2}, {y2})
            </td>
        </tr>
        """

    return f"""
    <div class="card">
        <h3>Detected Abnormalities (YOLOv8m)</h3>

        <table class="detection-table">
            <thead>
                <tr>
                    <th>Finding</th>
                    <th>Confidence</th>
                    <th>Bounding Box</th>
                </tr>
            </thead>

            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """


def _screen_updates(active: str):
    return [
        gr.update(visible=(name == active))
        for name in SCREEN_NAMES
    ]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Medical X-Ray Analysis") as demo:

    # -----------------------------------------------------------------------
    # Session state
    # -----------------------------------------------------------------------

    st_file_path = gr.State(None)

    st_prediction_id = gr.State(None)

    st_prediction_class = gr.State("")

    st_prediction_conf = gr.State(0.0)

    st_bboxes = gr.State([])

    st_img_original = gr.State(None)

    st_img_annotated = gr.State(None)

    st_img_heatmap = gr.State(None)

    # -----------------------------------------------------------------------
    # Backend health
    # -----------------------------------------------------------------------

    health = api_client.check_health()

    # =======================================================================
    # SCREEN 1 — WELCOME
    # =======================================================================

    with gr.Column(visible=True) as welcome_screen:

        gr.HTML(
            """
            <div class="welcome-hero">
                <div class="welcome-badge">AI-Assisted Radiology Support</div>
                <h1>Medical X-Ray Analysis</h1>

                <p>
                    Upload a chest X-ray and get an AI-assisted classification,
                    abnormality localization, and a visual explainability
                    heatmap — reviewed alongside your clinical judgment,
                    not in place of it.
                </p>

                <div class="feature-strip">
                    <div class="feature-pill">🩻&nbsp; ResNet-50 Classification</div>
                    <div class="feature-pill">🎯&nbsp; YOLOv8 Detection</div>
                    <div class="feature-pill">🔥&nbsp; Grad-CAM Explainability</div>
                </div>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="icon-circle blue">📤</div>
                    <h4>1. Upload</h4>
                    <p>Add a chest X-ray image in PNG or JPEG format, up to 10 MB.</p>
                </div>
                <div class="feature-card">
                    <div class="icon-circle teal">🧠</div>
                    <h4>2. AI Analysis</h4>
                    <p>The image runs through real ResNet-50 and YOLOv8 models on the backend.</p>
                </div>
                <div class="feature-card">
                    <div class="icon-circle coral">📄</div>
                    <h4>3. Review & Report</h4>
                    <p>View results, the Grad-CAM heatmap, and download a PDF summary.</p>
                </div>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="disclaimer-banner">
                ⚕️ This tool provides AI-assisted results for research and decision support —
                it is not a confirmed medical diagnosis.
            </div>
            """
        )

        if not health["reachable"]:
            gr.Markdown(
                f"""
                **Backend not reachable**

                Backend:
                `{api_client.BACKEND_URL}`

                Start it with:

                `uvicorn app.main:app`
                """
            )

        start_btn = gr.Button(
            "Start Analysis →",
            variant="primary",
            size="lg",
        )

    # =======================================================================
    # SCREEN 2 — PATIENT + X-RAY INPUT
    # =======================================================================

    with gr.Column(visible=False) as input_screen:

        gr.HTML(
            '<div class="screen-title">'
            "New X-Ray Analysis"
            "</div>"
        )

        gr.HTML(
            """
            <div class="card">
                <h3>Patient Information</h3>
            </div>
            """
        )

        with gr.Row():

            patient_name_in = gr.Textbox(
                label="Patient Name",
                placeholder="Optional",
            )

            patient_id_in = gr.Textbox(
                label="Patient ID",
                placeholder="Optional",
            )

        gr.HTML(
            """
            <div class="card">
                <h3>X-Ray Upload</h3>
                <p class="card-note">
                    PNG or JPEG, max 10 MB.
                </p>
            </div>
            """
        )

        file_input = gr.File(
            label="Upload X-Ray",
            file_types=[".png", ".jpg", ".jpeg"],
            type="filepath",
        )

        upload_preview = gr.Image(
            label="Preview",
            interactive=False,
            visible=False,
            height=240,
        )

        upload_status = gr.HTML(
            visible=False
        )

        with gr.Row():

            back_to_welcome_btn = gr.Button(
                "← Back"
            )

            continue_btn = gr.Button(
                "Continue →",
                variant="primary",
                interactive=False,
            )

    # =======================================================================
    # SCREEN 3 — ANALYSIS
    # =======================================================================

    with gr.Column(visible=False) as analysis_screen:

        gr.HTML(
            '<div class="screen-title">'
            "Analyzing X-Ray"
            "</div>"
        )

        checklist_out = gr.HTML(
            _checklist_html(0)
        )

        analysis_error = gr.Markdown(
            "",
            visible=False,
        )

    # =======================================================================
    # SCREEN 4 — RESULTS
    # =======================================================================

    with gr.Column(visible=False) as results_screen:

        gr.HTML(
            '<div class="screen-title">'
            "Analysis Results"
            "</div>"
        )

        prediction_html_out = gr.HTML()

        with gr.Row():

            with gr.Column(scale=1):

                classification_detail = gr.HTML()

            with gr.Column(scale=1):

                detections_html_out = gr.HTML()

        gr.HTML(
            """
            <div class="card">
                <h3>X-Ray Viewer</h3>
            </div>
            """
        )

        with gr.Row():

            view_original_btn = gr.Button(
                "Original"
            )

            view_annotated_btn = gr.Button(
                "Annotated"
            )

            view_gradcam_btn = gr.Button(
                "Grad-CAM"
            )

        viewer_image = gr.Image(
            interactive=False,
            height=380,
        )

        gradcam_note = gr.Markdown(
            visible=False
        )

        generate_report_btn = gr.Button(
            "Generate Medical Report →",
            variant="primary",
            size="lg",
        )

        results_status = gr.Markdown("")

    # =======================================================================
    # SCREEN 5 — REPORT
    # =======================================================================

    with gr.Column(visible=False) as report_screen:

        gr.HTML(
            '<div class="screen-title">'
            "Medical Analysis Report"
            "</div>"
        )

        report_status = gr.Markdown("")

        report_preview = gr.HTML(
            visible=False
        )

        with gr.Row():

            view_report_btn = gr.Button(
                "View Report"
            )

            report_file = gr.File(
                label="Download PDF",
                interactive=False,
            )

            new_analysis_btn = gr.Button(
                "New Analysis",
                variant="primary",
            )

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------

    gr.HTML(
        f"""
        <div class="disclaimer-footer">
            {FOOTER_DISCLAIMER}
        </div>
        """
    )

    # =======================================================================
    # INTERACTIONS
    # =======================================================================

    # -----------------------------------------------------------------------
    # Welcome -> Input
    # -----------------------------------------------------------------------

    start_btn.click(
        fn=lambda: _screen_updates("input"),
        outputs=[
            welcome_screen,
            input_screen,
            analysis_screen,
            results_screen,
            report_screen,
        ],
    )

    # -----------------------------------------------------------------------
    # Input: Upload + validation
    # -----------------------------------------------------------------------

    def on_upload(file_path):

        error = _validate_upload(file_path)

        if error:

            return (
                gr.update(
                    visible=False,
                    value=None,
                ),

                gr.update(
                    visible=True,
                    value=(
                        '<div class="upload-status-bad">'
                        f"⚠ {error}"
                        "</div>"
                    ),
                ),

                gr.update(
                    interactive=False
                ),

                None,
            )

        try:

            img = Image.open(file_path)

            w, h = img.size

            filename = os.path.basename(
                file_path
            )

            status_html = (
                '<div class="upload-status-ok">'
                f"✓ Valid X-ray image<br>"
                f"{filename}<br>"
                f"{w} × {h}"
                "</div>"
            )

            return (
                gr.update(
                    visible=True,
                    value=img,
                ),

                gr.update(
                    visible=True,
                    value=status_html,
                ),

                gr.update(
                    interactive=True
                ),

                file_path,
            )

        except Exception as exc:

            logger.exception(
                "Unable to read uploaded image"
            )

            return (
                gr.update(
                    visible=False,
                    value=None,
                ),

                gr.update(
                    visible=True,
                    value=(
                        '<div class="upload-status-bad">'
                        f"⚠ Unable to read image: {exc}"
                        "</div>"
                    ),
                ),

                gr.update(
                    interactive=False
                ),

                None,
            )

    file_input.change(
        fn=on_upload,
        inputs=[file_input],
        outputs=[
            upload_preview,
            upload_status,
            continue_btn,
            st_file_path,
        ],
    )

    # -----------------------------------------------------------------------
    # Input -> Welcome
    # -----------------------------------------------------------------------

    back_to_welcome_btn.click(
        fn=lambda: _screen_updates("welcome"),
        outputs=[
            welcome_screen,
            input_screen,
            analysis_screen,
            results_screen,
            report_screen,
        ],
    )

    # =======================================================================
    # INPUT -> ANALYSIS -> RESULTS
    # =======================================================================

    def run_pipeline(
        file_path,
        patient_name,
        patient_id_text,
    ):

        # ---------------------------------------------------------------
        # Immediately display Analysis screen
        # ---------------------------------------------------------------

        yield (
            *_screen_updates("analysis"),

            _checklist_html(0),

            "",  # prediction HTML
            "",  # classification
            "",  # detections

            None,  # viewer image

            None,  # state original
            None,  # state annotated
            None,  # state heatmap

            None,  # prediction ID
            "",    # prediction class
            0.0,   # confidence
            [],    # bboxes

            gr.update(visible=False),  # gradcam note
            gr.update(
                value="",
                visible=False,
            ),  # analysis error
        )

        try:

            # -----------------------------------------------------------
            # Validate once more before backend call
            # -----------------------------------------------------------

            validation_error = _validate_upload(
                file_path
            )

            if validation_error:

                raise ValueError(
                    validation_error
                )

            # -----------------------------------------------------------
            # Create patient
            # -----------------------------------------------------------

            display_name = None

            if (
                patient_name
                and patient_name.strip()
            ):

                display_name = (
                    patient_name.strip()
                )

                if (
                    patient_id_text
                    and patient_id_text.strip()
                ):

                    display_name = (
                        f"{display_name} "
                        f"(ID: {patient_id_text.strip()})"
                    )

            elif (
                patient_id_text
                and patient_id_text.strip()
            ):

                display_name = (
                    f"ID: {patient_id_text.strip()}"
                )

            patient_id = None

            if display_name:

                patient_id = (
                    api_client.create_patient(
                        display_name
                    )
                )

            # -----------------------------------------------------------
            # Upload image
            # -----------------------------------------------------------

            image_meta = api_client.upload_image(
                file_path,
                patient_id,
            )

            image_id = image_meta["id"]

            yield (
                *_screen_updates("analysis"),

                _checklist_html(1),

                "",
                "",
                "",

                None,

                None,
                None,
                None,

                None,
                "",
                0.0,
                [],

                gr.update(
                    visible=False
                ),

                gr.update(
                    value="",
                    visible=False,
                ),
            )

            # -----------------------------------------------------------
            # Server-side preprocessing
            # -----------------------------------------------------------

            yield (
                *_screen_updates("analysis"),

                _checklist_html(2),

                "",
                "",
                "",

                None,

                None,
                None,
                None,

                None,
                "",
                0.0,
                [],

                gr.update(
                    visible=False
                ),

                gr.update(
                    value="",
                    visible=False,
                ),
            )

            # -----------------------------------------------------------
            # REAL prediction
            # -----------------------------------------------------------

            prediction = (
                api_client.create_prediction(
                    image_id
                )
            )

            # -----------------------------------------------------------
            # IMPORTANT:
            # Capture the real prediction ID
            # -----------------------------------------------------------

            prediction_id = (
                prediction.get("id")
                or prediction.get("prediction_id")
            )

            if prediction_id is None:

                raise ValueError(
                    "Prediction was returned by the backend, "
                    "but no prediction ID was found."
                )

            yield (
                *_screen_updates("analysis"),

                _checklist_html(3),

                "",
                "",
                "",

                None,

                None,
                None,
                None,

                prediction_id,
                "",
                0.0,
                [],

                gr.update(
                    visible=False
                ),

                gr.update(
                    value="",
                    visible=False,
                ),
            )

            # -----------------------------------------------------------
            # YOLO detections
            # -----------------------------------------------------------

            bboxes = prediction.get(
                "bboxes",
                []
            )

            yield (
                *_screen_updates("analysis"),

                _checklist_html(4),

                "",
                "",
                "",

                None,

                None,
                None,
                None,

                prediction_id,
                "",
                0.0,
                bboxes,

                gr.update(
                    visible=False
                ),

                gr.update(
                    value="",
                    visible=False,
                ),
            )

            # -----------------------------------------------------------
            # Grad-CAM
            # -----------------------------------------------------------

            heatmap_path = prediction.get(
                "heatmap_path"
            )

            heatmap_image = None

            if heatmap_path:

                heatmap_bytes = (
                    api_client.fetch_file_bytes(
                        heatmap_path
                    )
                )

                heatmap_image = Image.open(
                    io.BytesIO(heatmap_bytes)
                )

            if heatmap_image:

                gradcam_note_update = gr.update(
                    visible=True,
                    value=(
                        "_Grad-CAM highlights image "
                        "regions that contributed to the "
                        "ResNet-50 classification output. "
                        "It does not prove a disease location._"
                    ),
                )

            else:

                gradcam_note_update = gr.update(
                    visible=False
                )

            # -----------------------------------------------------------
            # Images
            # -----------------------------------------------------------

            original_image = Image.open(
                file_path
            )

            annotated_image = draw_bboxes(
                original_image,
                bboxes,
            )

            # -----------------------------------------------------------
            # Classification
            # -----------------------------------------------------------

            predicted_class = prediction[
                "predicted_class"
            ]

            confidence = float(
                prediction["confidence"]
            )

            # -----------------------------------------------------------
            # Build result UI
            # -----------------------------------------------------------

            prediction_html = (
                _prediction_card_html(
                    predicted_class,
                    confidence,
                )
            )

            classification_html = (
                '<div class="card">'
                "<h3>Classification (ResNet-50)</h3>"
                f"<p>Predicted Class: "
                f"<b>{predicted_class}</b></p>"
                f"<p>Confidence: "
                f"<b>{confidence * 100:.1f}%</b></p>"
                "</div>"
            )

            detections_html = (
                _detections_table_html(
                    bboxes
                )
            )

            # -----------------------------------------------------------
            # Final Results screen
            # -----------------------------------------------------------

            yield (
                *_screen_updates("results"),

                _checklist_html(5),

                prediction_html,
                classification_html,
                detections_html,

                original_image,

                original_image,
                annotated_image,
                heatmap_image,

                prediction_id,
                predicted_class,
                confidence,
                bboxes,

                gradcam_note_update,

                gr.update(
                    value="",
                    visible=False,
                ),
            )

        except Exception as exc:

            err = _human_error(exc)

            logger.error(
                "Pipeline failed: %s",
                err,
            )

            # Return user to input screen and SHOW the error.
            yield (
                *_screen_updates("input"),

                _checklist_html(0),

                "",
                "",
                "",

                None,

                None,
                None,
                None,

                None,
                "",
                0.0,
                [],

                gr.update(
                    visible=False
                ),

                gr.update(
                    value=err,
                    visible=True,
                ),
            )

    continue_btn.click(
        fn=run_pipeline,
        inputs=[
            st_file_path,
            patient_name_in,
            patient_id_in,
        ],
        outputs=[
            welcome_screen,
            input_screen,
            analysis_screen,
            results_screen,
            report_screen,

            checklist_out,

            prediction_html_out,
            classification_detail,
            detections_html_out,

            viewer_image,

            st_img_original,
            st_img_annotated,
            st_img_heatmap,

            st_prediction_id,
            st_prediction_class,
            st_prediction_conf,
            st_bboxes,

            gradcam_note,

            analysis_error,
        ],
    )

    # =======================================================================
    # IMAGE VIEWER
    # =======================================================================

    view_original_btn.click(
        fn=lambda img: img,
        inputs=[st_img_original],
        outputs=[viewer_image],
    )

    view_annotated_btn.click(
        fn=lambda img: img,
        inputs=[st_img_annotated],
        outputs=[viewer_image],
    )

    view_gradcam_btn.click(
        fn=lambda img: img,
        inputs=[st_img_heatmap],
        outputs=[viewer_image],
    )

    # =======================================================================
    # RESULTS -> REPORT
    # =======================================================================

    def do_generate_report(
        prediction_id
    ):

        if not prediction_id:

            return (
                *_screen_updates("report"),

                "⚠️ No analysis available to report on.",

                gr.update(
                    visible=False
                ),

                None,
            )

        try:

            logger.info(
                "Generating report for prediction %s",
                prediction_id,
            )

            report = (
                api_client.create_report(
                    int(prediction_id)
                )
            )

            pdf_path = report.get(
                "pdf_path"
            )

            if not pdf_path:

                return (
                    *_screen_updates("report"),

                    "⚠️ Report was created but "
                    "no PDF was produced.",

                    gr.update(
                        visible=False
                    ),

                    None,
                )

            # -----------------------------------------------------------
            # Download PDF bytes from backend
            # -----------------------------------------------------------

            pdf_bytes = (
                api_client.fetch_file_bytes(
                    pdf_path
                )
            )

            # -----------------------------------------------------------
            # Save temporary local copy for Gradio download
            # -----------------------------------------------------------

            out_path = os.path.join(
                tempfile.gettempdir(),
                f"xray_report_{prediction_id}.pdf",
            )

            with open(
                out_path,
                "wb",
            ) as f:

                f.write(pdf_bytes)

            # -----------------------------------------------------------
            # PDF preview
            # -----------------------------------------------------------

            b64 = base64.b64encode(
                pdf_bytes
            ).decode()

            preview_html = (
                f'<iframe '
                f'src="data:application/pdf;base64,{b64}" '
                'style="'
                'width:100%;'
                'height:560px;'
                'border:1px solid var(--line);'
                'border-radius:10px;'
                '">'
                "</iframe>"
            )

            return (
                *_screen_updates("report"),

                "✅ Report generated successfully.",

                gr.update(
                    visible=True,
                    value=preview_html,
                ),

                out_path,
            )

        except Exception as exc:

            err = _human_error(exc)

            return (
                *_screen_updates("report"),

                err,

                gr.update(
                    visible=False
                ),

                None,
            )

    generate_report_btn.click(
        fn=do_generate_report,
        inputs=[
            st_prediction_id
        ],
        outputs=[
            welcome_screen,
            input_screen,
            analysis_screen,
            results_screen,
            report_screen,

            report_status,
            report_preview,
            report_file,
        ],
    )

    # =======================================================================
    # VIEW REPORT
    # =======================================================================

    def show_report(
        current_report
    ):

        if current_report:

            return gr.update(
                visible=True
            )

        return gr.update()

    view_report_btn.click(
        fn=show_report,
        inputs=[
            report_preview
        ],
        outputs=[
            report_preview
        ],
    )

    # =======================================================================
    # NEW ANALYSIS / RESET
    # =======================================================================

    def new_analysis():

        return (
            # Screens
            *_screen_updates("input"),

            # Input
            None,
            "",
            "",

            # Upload UI
            gr.update(
                visible=False,
                value=None,
            ),

            gr.update(
                visible=False,
                value="",
            ),

            gr.update(
                interactive=False
            ),

            # State
            None,       # file path
            None,       # prediction ID
            "",         # prediction class
            0.0,        # confidence
            [],         # bboxes
            None,       # original
            None,       # annotated
            None,       # heatmap

            # Result UI
            "",
            "",
            "",
            None,

            # Grad-CAM note
            gr.update(
                visible=False,
                value="",
            ),

            # Report
            "",
            gr.update(
                visible=False,
                value="",
            ),
            None,

            # Analysis error
            gr.update(
                visible=False,
                value="",
            ),
        )

    new_analysis_btn.click(
        fn=new_analysis,
        outputs=[
            welcome_screen,
            input_screen,
            analysis_screen,
            results_screen,
            report_screen,

            file_input,
            patient_name_in,
            patient_id_in,

            upload_preview,
            upload_status,
            continue_btn,

            st_file_path,
            st_prediction_id,
            st_prediction_class,
            st_prediction_conf,
            st_bboxes,

            st_img_original,
            st_img_annotated,
            st_img_heatmap,

            prediction_html_out,
            classification_detail,
            detections_html_out,

            viewer_image,
            gradcam_note,

            report_status,
            report_preview,
            report_file,

            analysis_error,
        ],
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    demo.queue().launch(
        theme=gr.themes.Soft(
            primary_hue="blue",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
    )