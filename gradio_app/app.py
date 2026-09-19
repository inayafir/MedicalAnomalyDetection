"""
Gradio prototype UI for the Medical X-Ray Anomaly Detection project.

This is the official demonstration interface named in the project synopsis.
It is a thin client over the existing FastAPI backend — every classification,
detection, confidence value, and Grad-CAM heatmap shown here comes from a
real call to the backend's real ResNet-50 / YOLOv8m / Grad-CAM pipeline.
Nothing is fabricated or mocked here.

Run with:  python gradio_app/app.py
Requires the backend running separately (see README): uvicorn app.main:app
"""
from __future__ import annotations

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gradio_app")

DISCLAIMER = (
    "**This system is an academic/research prototype** intended for decision "
    "support and demonstration purposes. It is **not a medical diagnosis** "
    "and should not replace evaluation by a qualified healthcare professional."
)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_FILE_MB = 10


def _human_error(exc: Exception) -> str:
    """Converts any exception into a short, non-technical message. Full detail
    is always logged server-side for debugging, never shown to the user."""
    logger.exception("Operation failed")
    if isinstance(exc, api_client.BackendError):
        return f"⚠️ {exc}"
    return "⚠️ Something went wrong. Please try again or check the backend logs."


def _validate_upload(file_path: str | None) -> str | None:
    """Returns an error message, or None if the file is valid."""
    if not file_path:
        return "Please upload an X-ray image first."
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"Unsupported file format '{ext}'. Please use PNG or JPEG."
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return f"File is {size_mb:.1f} MB — please upload an image under {MAX_FILE_MB} MB."
    return None


def run_analysis(file_path, patient_name, progress=gr.Progress()):
    """
    Full pipeline: upload -> real ResNet-50 classification -> real YOLOv8m
    detection -> real Grad-CAM -> results. Yields progressively so the UI
    shows a genuine multi-stage status instead of a single spinner.
    """
    empty_state = (None, None, None, None, None, None, None)

    error = _validate_upload(file_path)
    if error:
        yield ("", error, *empty_state[2:])
        return

    try:
        progress(0.1, desc="Uploading image...")
        patient_id = None
        if patient_name and patient_name.strip():
            patient_id = api_client.create_patient(patient_name.strip())

        image_meta = api_client.upload_image(file_path, patient_id)
        image_id = image_meta["id"]

        progress(0.35, desc="Running ResNet-50 classification and YOLOv8m detection...")
        prediction = api_client.create_prediction(image_id)

        progress(0.75, desc="Generating Grad-CAM explanation...")
        heatmap_path = prediction.get("heatmap_path")
        heatmap_image = None
        if heatmap_path:
            heatmap_bytes = api_client.fetch_file_bytes(heatmap_path)
            heatmap_image = Image.open(io.BytesIO(heatmap_bytes))

        progress(0.9, desc="Preparing results...")
        original_image = Image.open(file_path)
        bboxes = prediction.get("bboxes", [])
        annotated_image = draw_bboxes(original_image, bboxes)

        predicted_class = prediction["predicted_class"]
        confidence = prediction["confidence"]
        classification_md = (
            f"### {predicted_class}\n"
            f"**Confidence: {confidence * 100:.1f}%**\n\n"
            f"_(ResNet-50, 15-class chest X-ray classifier — this is a model "
            f"output, not a clinical diagnosis)_"
        )

        if bboxes:
            table_rows = [
                [b.get("class") or b.get("class_", ""), f"{b.get('confidence', 0) * 100:.1f}%",
                 b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2")]
                for b in bboxes
            ]
            detections_md = f"**{len(bboxes)} abnormality region(s) detected** (YOLOv8m)"
        else:
            table_rows = []
            detections_md = "**No abnormality regions detected** by YOLOv8m at the current threshold."

        progress(1.0, desc="Done")
        status = f"✅ Analysis complete — prediction id {prediction['id']}"

        yield (
            status,
            classification_md,
            detections_md,
            table_rows,
            original_image,
            annotated_image,
            heatmap_image,
            prediction["id"],
        )
    except Exception as exc:
        yield (_human_error(exc), "", "", [], None, None, None, None)


def generate_report(prediction_id):
    if not prediction_id:
        return None, "⚠️ Run an analysis first before generating a report."
    try:
        report = api_client.create_report(int(prediction_id))
        pdf_path = report.get("pdf_path")
        if not pdf_path:
            return None, "⚠️ Report was created but no PDF was produced. Check backend logs."
        pdf_bytes = api_client.fetch_file_bytes(pdf_path)
        out_path = os.path.join(tempfile.gettempdir(), f"xray_report_{prediction_id}.pdf")
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)
        return out_path, "✅ Report generated. Click below to download."
    except Exception as exc:
        return None, _human_error(exc)


def reset_workflow():
    return (
        None, "",  # file, patient_name
        "", "", "", [], None, None, None, None,  # status, classification, detections_md, table, imgs, pred_id
        None, "",  # report file, report status
    )


CUSTOM_CSS = """
.disclaimer-box { border: 1px solid var(--border-color-primary); border-radius: 8px;
    padding: 10px 14px; font-size: 0.85em; background: var(--background-fill-secondary); }
footer { visibility: hidden }
"""

with gr.Blocks(title="Medical X-Ray Anomaly Detection") as demo:
    prediction_id_state = gr.State(None)

    gr.Markdown(
        "# 🩻 Medical X-Ray Anomaly Detection\n"
        "AI-assisted chest X-ray analysis using **ResNet-50 classification**, "
        "**YOLOv8m abnormality detection**, and **Grad-CAM explainability**.\n\n"
        "_Academic research prototype — official demonstration interface._"
    )
    gr.Markdown(DISCLAIMER, elem_classes=["disclaimer-box"])

    health = api_client.check_health()
    if not health["reachable"]:
        gr.Markdown(
            f"🔴 **Backend not reachable** at `{api_client.BACKEND_URL}`. "
            "Start it with `uvicorn app.main:app` before analyzing images."
        )
    elif not health["model_loaded"]:
        gr.Markdown(
            "🟡 **Backend is reachable, but models are not loaded.** "
            "Check that the real checkpoint files are in place (see README)."
        )
    else:
        gr.Markdown("🟢 **Backend connected — real ResNet-50 and YOLOv8m models loaded.**")

    with gr.Tab("Analyze"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Step 1 — Upload X-Ray")
                file_input = gr.File(
                    label="Chest X-Ray (PNG or JPEG, max 10 MB)",
                    file_types=[".png", ".jpg", ".jpeg"],
                    type="filepath",
                )
                gr.Markdown("### Step 2 — Patient Information (optional)")
                patient_name = gr.Textbox(
                    label="Patient name/ID (optional)",
                    placeholder="Leave blank to analyze anonymously",
                )
                gr.Markdown("### Step 3 — Analyze")
                analyze_btn = gr.Button("🔬 Analyze X-Ray", variant="primary", size="lg")
                status_box = gr.Markdown("")
                reset_btn = gr.Button("↺ Reset", size="sm")

            with gr.Column(scale=2):
                gr.Markdown("### Results")
                with gr.Row():
                    classification_out = gr.Markdown(label="Classification")
                    detections_out = gr.Markdown(label="Detections")

                with gr.Tabs():
                    with gr.Tab("Original"):
                        original_out = gr.Image(label="Original X-Ray", interactive=False)
                    with gr.Tab("Detections (YOLOv8m)"):
                        annotated_out = gr.Image(label="Annotated X-Ray", interactive=False)
                    with gr.Tab("Grad-CAM"):
                        heatmap_out = gr.Image(label="Grad-CAM Heatmap", interactive=False)

                detections_table = gr.Dataframe(
                    headers=["Class", "Confidence", "X1", "Y1", "X2", "Y2"],
                    label="Detection Details",
                    row_count=(0, "dynamic"),
                )

                gr.Markdown("### Step 4 — Report")
                with gr.Row():
                    report_btn = gr.Button("📄 Generate Report")
                    report_status = gr.Markdown("")
                report_file = gr.File(label="Download PDF Report", interactive=False)

        analyze_btn.click(
            fn=run_analysis,
            inputs=[file_input, patient_name],
            outputs=[
                status_box,
                classification_out,
                detections_out,
                detections_table,
                original_out,
                annotated_out,
                heatmap_out,
                prediction_id_state,
            ],
        )

        report_btn.click(
            fn=generate_report,
            inputs=[prediction_id_state],
            outputs=[report_file, report_status],
        )

        reset_btn.click(
            fn=reset_workflow,
            outputs=[
                file_input, patient_name,
                status_box, classification_out, detections_out, detections_table,
                original_out, annotated_out, heatmap_out, prediction_id_state,
                report_file, report_status,
            ],
        )

    with gr.Tab("About"):
        gr.Markdown(
            "## How this system works\n"
            "- **ResNet-50** — classifies the X-ray into one of 15 categories "
            "(14 abnormality types + Normal), trained on chest X-ray data.\n"
            "- **YOLOv8m** — localizes abnormality regions with bounding boxes, "
            "trained on 14 abnormality classes.\n"
            "- **Grad-CAM** — highlights the image regions that most influenced "
            "the classifier's decision, for explainability.\n\n"
            "## Model details\n"
            "- Classifier: 15 classes (14 findings + Normal)\n"
            "- Detector: 14 abnormality classes\n"
            "- Both are the project's own trained checkpoints, not off-the-shelf "
            "pretrained weights.\n\n"
            + DISCLAIMER
        )

if __name__ == "__main__":
    demo.queue().launch(
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CUSTOM_CSS,
    )