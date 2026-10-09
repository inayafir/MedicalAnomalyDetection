"""
Medical X-Ray Analysis - AI-assisted radiology support (Gradio, light/white UI)

THREE PAGES
  1. Landing : hero, 3 steps, disclaimer, "Start Analysis"
  2. Upload  : patient info (optional), X-ray upload, preview, validation, Back / Continue
  3. Report  : embedded PDF viewer, View Report / Download PDF / New Analysis

Pipeline: ResNet-50 classification -> YOLOv8m detection -> Grad-CAM -> PDF report

Run:  python app.py   ->  http://127.0.0.1:7860
"""

# IMPORTANT: torch and ultralytics must be imported FIRST on Windows.
# Importing cv2 / numpy / gradio before torch can cause
# "WinError 1114 ... c10.dll" errors.
import torch
from torchvision import models, transforms
from ultralytics import YOLO

import base64
import inspect
import io
import os
import tempfile
from datetime import datetime, timezone

import cv2
import gradio as gr
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
CKPT_DIR = r"C:\Users\kruth\OneDrive\Desktop\MedicalAnomalyDetection-main\backend\ml_core\checkpoints"
RESNET_WEIGHTS = os.path.join(CKPT_DIR, "resnet50.pth")
YOLO_WEIGHTS = os.path.join(CKPT_DIR, "yolov8m_14class.pt")

# Your trained ResNet has 15 outputs. Paste the 15 class names here, in the
# exact training order, e.g. ["Normal", "Pneumonia", ...].
# If left as None, names are read from the checkpoint if it stores them,
# otherwise shown as "Class 0", "Class 1", ...
CLASS_NAMES_OVERRIDE = None

MAX_FILE_MB = 10
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "xray_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DISCLAIMER = ("AI-generated results are intended to assist clinical review and "
              "should be interpreted by a qualified healthcare professional.")

# --------------------------------------------------------------------------
# MODELS (lazy-loaded)
# --------------------------------------------------------------------------
_models = {}
_class_names = []


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def get_resnet():
    if "resnet" not in _models:
        if not os.path.exists(RESNET_WEIGHTS):
            raise FileNotFoundError(f"ResNet-50 weights not found at '{RESNET_WEIGHTS}'.")
        ckpt = _load_checkpoint(RESNET_WEIGHTS)

        names = None
        state = ckpt
        if isinstance(ckpt, dict):
            for key in ("classes", "class_names", "labels"):
                if key in ckpt and isinstance(ckpt[key], (list, tuple)):
                    names = list(ckpt[key])
            for key in ("state_dict", "model_state_dict", "model"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    break
        if not isinstance(state, dict):
            raise RuntimeError("Unsupported ResNet checkpoint format (expected a state_dict).")
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

        n_cls = int(state["fc.weight"].shape[0])
        m = models.resnet50(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, n_cls)
        m.load_state_dict(state)
        _models["resnet"] = m.to(_device()).eval()

        if CLASS_NAMES_OVERRIDE and len(CLASS_NAMES_OVERRIDE) == n_cls:
            names = list(CLASS_NAMES_OVERRIDE)
        if not names or len(names) != n_cls:
            names = [f"Class {i}" for i in range(n_cls)]
        _class_names[:] = names
    return _models["resnet"]


def get_yolo():
    if "yolo" not in _models:
        if not os.path.exists(YOLO_WEIGHTS):
            raise FileNotFoundError(f"YOLOv8m weights not found at '{YOLO_WEIGHTS}'.")
        _models["yolo"] = YOLO(YOLO_WEIGHTS)
    return _models["yolo"]


# --------------------------------------------------------------------------
# INFERENCE
# --------------------------------------------------------------------------
def classify(img):
    model = get_resnet()
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(img.convert("RGB")).unsqueeze(0).to(_device())
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return idx, float(probs[idx]), x


def grad_cam(x, class_idx, original):
    model = get_resnet()
    target = model.layer4[-1]
    acts, grads = [], []
    h1 = target.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = target.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    try:
        x = x.clone().requires_grad_(True)
        model.zero_grad()
        model(x)[0, class_idx].backward()
    finally:
        h1.remove()
        h2.remove()
    w = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts[0]).sum(dim=1))[0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    base = np.array(original.convert("RGB"))
    cam = cv2.resize(cam, (base.shape[1], base.shape[0]))
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET),
                        cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(base, 0.55, heat, 0.45, 0))


def detect(img):
    res = get_yolo().predict(np.array(img.convert("RGB")), conf=0.01,
                             verbose=False)[0]
    rows = []
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        rows.append({"region": res.names[int(b.cls)], "conf": float(b.conf),
                     "box": (x1, y1, x2, y2)})
    rows.sort(key=lambda r: r["conf"], reverse=True)
    return rows


# --------------------------------------------------------------------------
# PDF REPORT
# --------------------------------------------------------------------------
def build_pdf(path, patient_id, patient_name, filename, original, cam_img,
              label, conf, regions):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image as RLImage, Paragraph,
                                    SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    TEAL = colors.HexColor("#0e6f8e")
    ss = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=20, spaceAfter=2)
    small = ParagraphStyle("s", parent=ss["Normal"], fontSize=8.5,
                           textColor=colors.HexColor("#555555"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=TEAL, fontSize=13)
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def footer(c, doc):
        c.saveState()
        c.setStrokeColor(colors.HexColor("#cccccc"))
        c.line(18 * mm, 16 * mm, A4[0] - 18 * mm, 16 * mm)
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawString(18 * mm, 11 * mm, DISCLAIMER)
        c.drawRightString(A4[0] - 18 * mm, 11 * mm, f"Page {doc.page}")
        c.restoreState()

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=18 * mm,
                            rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=22 * mm)
    s = [Paragraph("Chest X-Ray Analysis Report", title),
         Paragraph(f"Generated {now}", small), Spacer(1, 8)]

    info = Table([["Patient ID", patient_id or "N/A"],
                  ["Patient Name", patient_name or "N/A"],
                  ["Study Image", filename], ["Study Date", now],
                  ["Examination", "Chest X-Ray"]], colWidths=[40 * mm, 120 * mm])
    info.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    s += [info, Spacer(1, 10)]

    banner = Table([
        [Paragraph("<font color='white' size=8>AI ANALYSIS &nbsp;·&nbsp; PRIMARY CLASSIFICATION</font>", body)],
        [Paragraph(f"<font color='white' size=22><b>{label.upper()}</b></font>", body)],
        [Paragraph(f"<font color='white' size=9>Confidence: {conf * 100:.1f}% &nbsp;—&nbsp; Model: ResNet-50</font>", body)],
    ], colWidths=[160 * mm])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TEAL),
                                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                                ("TOPPADDING", (0, 0), (-1, -1), 6),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    s += [banner, Spacer(1, 10), Paragraph("Detected Regions (YOLOv8m)", h2),
          Paragraph("Localized abnormality candidates, independent of the primary "
                    "classification above. Low-confidence entries are not "
                    "confirmed findings.", small), Spacer(1, 4)]

    data = [["Region", "Confidence", "X1", "Y1", "X2", "Y2"]]
    for r in regions[:8]:
        data.append([r["region"], f"{r['conf'] * 100:.1f}%", *r["box"]])
    if len(data) == 1:
        data.append(["No regions detected", "-", "-", "-", "-", "-"])
    t = Table(data, colWidths=[50 * mm, 30 * mm, 20 * mm, 20 * mm, 20 * mm, 20 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6f2f7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), TEAL),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd"))]))
    s += [t, Spacer(1, 12), Paragraph("Image Analysis", h2)]

    def to_rl(im, w=78 * mm):
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "PNG")
        buf.seek(0)
        return RLImage(buf, width=w, height=w * im.height / im.width)

    imgs = Table([[to_rl(original), to_rl(cam_img)],
                  [Paragraph("Original X-Ray", small),
                   Paragraph("Grad-CAM Explainability", small)]],
                 colWidths=[82 * mm, 82 * mm])
    imgs.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    s += [imgs, Spacer(1, 6),
          Paragraph("Grad-CAM highlights the image regions that most influenced "
                    "the ResNet-50 classification decision.", small)]
    doc.build(s, onFirstPage=footer, onLaterPages=footer)


# --------------------------------------------------------------------------
# HELPERS / HANDLERS
# --------------------------------------------------------------------------
def _status(msg, ok):
    return f"<div class='status {'ok' if ok else 'bad'}'><b>{'✓' if ok else '✕'} {msg}</b></div>"


def validate_upload(path):
    if not path:
        return None, "", False
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg"):
        return None, _status("Only PNG or JPEG images are supported.", False), False
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return None, _status(f"File is {size_mb:.1f} MB. Max is {MAX_FILE_MB} MB.", False), False
    try:
        img = Image.open(path)
        img.load()
    except Exception:
        return None, _status("Could not read this file as an image.", False), False
    w, h = img.size
    if min(w, h) < 64:
        return None, _status(f"Image too small ({w} × {h}).", False), False
    html = (f"<div class='status ok'><b>✓ Valid X-ray image</b><br>"
            f"<b>{os.path.basename(path)}</b><br><b>{w} × {h}</b></div>")
    return img.convert("RGB"), html, True


def pdf_viewer_html(pdf_path):
    with open(pdf_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (f"<iframe src='data:application/pdf;base64,{b64}' "
            f"style='width:100%;height:680px;border:0;border-radius:12px;"
            f"background:#2b2b2b'></iframe>")


def show(screen):
    return [gr.update(visible=(screen == s)) for s in ("landing", "upload", "report")]


def on_file(path):
    img, html, ok = validate_upload(path)
    return (gr.update(value=img, visible=img is not None),
            gr.update(value=html, visible=bool(html)),
            gr.update(interactive=ok))


def clear_all():
    return (None, "", "", gr.update(value=None, visible=False),
            gr.update(value="", visible=False), gr.update(interactive=False))


def run_analysis(name, pid, path, progress=gr.Progress()):
    try:
        progress(0.05, desc="Reading image…")
        original = Image.open(path).convert("RGB")
        progress(0.25, desc="Classifying with ResNet-50…")
        idx, conf, x = classify(original)
        progress(0.50, desc="Detecting regions with YOLOv8m…")
        regions = detect(original)
        progress(0.75, desc="Generating Grad-CAM…")
        cam_img = grad_cam(x, idx, original)
        progress(0.90, desc="Building PDF report…")
        out = os.path.join(OUTPUT_DIR, f"xray_report_{datetime.now():%Y%m%d_%H%M%S}.pdf")
        build_pdf(out, (pid or "").strip(), (name or "").strip(),
                  os.path.basename(path), original, cam_img,
                  _class_names[idx], conf, regions)
        size_kb = os.path.getsize(out) / 1024
        return (*show("report"), pdf_viewer_html(out),
                gr.update(value=out, label=f"Download PDF · {size_kb:.1f} KB"),
                gr.update(value="", visible=False))
    except Exception as e:
        err = f"<div class='status bad'><b>✕ Analysis failed</b><br>{e}</div>"
        return (*show("upload"), "", gr.update(), gr.update(value=err, visible=True))


# --------------------------------------------------------------------------
# STYLE (white background, light mode forced)
# --------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
* { font-family:'Montserrat',sans-serif !important; }
body, .gradio-container, .gradio-container.dark { background:#ffffff !important; color:#1f2937 !important; }
.gradio-container { max-width:1440px !important; margin:auto; }
footer { display:none !important; }
.block, .form, .panel { background:#ffffff !important; border-color:#e5e9f0 !important; }
.gradio-container label span, .gradio-container .label-wrap span { color:#475569 !important; }

.page-title { padding:14px 14px; }
.page-title h1 { color:#0b3d91; font-size:28px; font-weight:800; margin:0; }
.section { border:1px solid #e3e8ef; border-radius:14px; padding:18px 22px; background:#fff; }
.section h3 { color:#0b3d91; font-size:16px; font-weight:700; margin:0 0 4px; }
.section p { color:#6b7a90; font-size:13px; margin:0; }

.hero { background:linear-gradient(120deg,#0a3a8a 0%,#0b4f9e 45%,#10a5a5 100%);
  border-radius:24px; padding:48px 40px 40px; text-align:center;
  box-shadow:0 18px 40px rgba(10,58,138,.22); }
.hero .badge { display:inline-block; padding:7px 22px; border-radius:999px;
  background:rgba(255,255,255,.92); color:#0b3d91; font-weight:700; font-size:12px;
  letter-spacing:.8px; border:1px solid #cfe0f5; }
.hero h1 { color:#ffffff; font-size:40px; font-weight:800; margin:26px 0 12px; }
.hero p { color:rgba(255,255,255,.85); max-width:660px; margin:0 auto 26px; font-size:16px; line-height:1.55; }
.chips { display:flex; gap:14px; justify-content:center; flex-wrap:wrap; }
.chip { background:rgba(255,255,255,.95); color:#0b3d91; font-weight:700; font-size:14px;
  padding:12px 22px; border-radius:12px; border:1px solid #cfe0f5; }

.steps { display:grid; grid-template-columns:repeat(3,1fr); gap:18px; }
.step { border:1px solid #e3e8ef; border-radius:18px; padding:24px 20px; text-align:center; background:#fff; }
.step .ico { width:56px; height:56px; border-radius:14px; margin:0 auto 12px;
  display:flex; align-items:center; justify-content:center; font-size:26px; }
.step h4 { color:#0b3d91; font-size:17px; font-weight:700; margin:6px 0; }
.step p { color:#6b7a90; font-size:13px; line-height:1.5; margin:0; }
.i1 { background:#dbeafe; } .i2 { background:#c9f5e6; } .i3 { background:#fde0e6; }

.notice { background:#fffbea; border:1px solid #f4e3a1; color:#7a5b00;
  border-radius:12px; padding:14px; text-align:center; font-size:14px; }

.btn-primary button, button.btn-primary { background:#2563eb !important; color:#fff !important;
  border:none !important; border-radius:6px !important; font-weight:700 !important;
  font-size:17px !important; height:48px; }
.btn-primary button:hover { background:#1d4fd8 !important; }
.btn-primary button:disabled, .btn-primary button[disabled] { background:#a9c1ee !important; opacity:1 !important; }
.btn-dark button, button.btn-dark { background:#475569 !important; color:#fff !important;
  border:none !important; border-radius:6px !important; font-weight:700 !important;
  font-size:17px !important; height:48px; }
.btn-dark button:hover { background:#3b4859 !important; }

.status { border-radius:12px; padding:16px 18px; font-size:15px; line-height:1.5; }
.status.ok { background:#e8fbef; color:#15803d; }
.status.bad { background:#fdecec; color:#b42318; }
.disclaimer { border-top:1px solid #e3e8ef; margin-top:10px; padding-top:16px;
  text-align:center; color:#6b7a90; font-size:13px; }
.success { padding:6px 10px; font-size:15px; color:#1f2937; }
"""

FORCE_LIGHT_JS = """
() => {
  document.body.classList.remove('dark');
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'light') {
    u.searchParams.set('__theme', 'light');
    window.location.replace(u);
  }
}
"""

LANDING_HERO = """
<div class="hero">
  <span class="badge">AI-ASSISTED RADIOLOGY SUPPORT</span>
  <h1>Medical X-Ray Analysis</h1>
  <p>Upload a chest X-ray and get an AI-assisted classification, abnormality
  localization, and a visual explainability heatmap — reviewed alongside your
  clinical judgment, not in place of it.</p>
  <div class="chips">
    <span class="chip">🩻 ResNet-50 Classification</span>
    <span class="chip">🎯 YOLOv8 Detection</span>
    <span class="chip">🔥 Grad-CAM Explainability</span>
  </div>
</div>
"""

STEPS = """
<div class="steps">
  <div class="step"><div class="ico i1">📤</div><h4>1. Upload</h4>
    <p>Add a chest X-ray image in PNG or JPEG format, up to 10 MB.</p></div>
  <div class="step"><div class="ico i2">🧠</div><h4>2. AI Analysis</h4>
    <p>The image runs through real ResNet-50 and YOLOv8 models.</p></div>
  <div class="step"><div class="ico i3">📄</div><h4>3. Review &amp; Report</h4>
    <p>View results, the Grad-CAM heatmap, and download a PDF summary.</p></div>
</div>
"""

THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate").set(
    body_background_fill="#ffffff", body_background_fill_dark="#ffffff",
    block_background_fill="#ffffff", block_background_fill_dark="#ffffff",
    body_text_color="#1f2937", body_text_color_dark="#1f2937",
    input_background_fill="#fbfcfe", input_background_fill_dark="#fbfcfe",
)

# --------------------------------------------------------------------------
# APP
# --------------------------------------------------------------------------
# Gradio 6 moved theme/css/js from Blocks() to launch(); support both.
_NEW_API = "css" in inspect.signature(gr.Blocks.launch).parameters
_BLOCKS_KW = {} if _NEW_API else dict(theme=THEME, css=CSS, js=FORCE_LIGHT_JS)
_LAUNCH_KW = dict(theme=THEME, css=CSS, js=FORCE_LIGHT_JS) if _NEW_API else {}

with gr.Blocks(title="Medical X-Ray Analysis", **_BLOCKS_KW) as demo:

    # ===== PAGE 1: Landing =====
    with gr.Column(visible=True) as landing:
        gr.HTML(LANDING_HERO)
        gr.HTML(STEPS)
        gr.HTML("<div class='notice'>⚕️ This tool provides AI-assisted results "
                "for research and decision support — it is not a confirmed "
                "medical diagnosis.</div>")
        start_btn = gr.Button("Start Analysis →", elem_classes="btn-primary")

    # ===== PAGE 2: Upload =====
    with gr.Column(visible=False) as upload:
        gr.HTML("<div class='page-title'><h1>New X-Ray Analysis</h1></div>")
        gr.HTML("<div class='section'><h3>Patient Information</h3></div>")
        with gr.Row():
            p_name = gr.Textbox(label="Patient Name", placeholder="Optional")
            p_id = gr.Textbox(label="Patient ID", placeholder="Optional")
        gr.HTML("<div class='section'><h3>X-Ray Upload</h3>"
                "<p>PNG or JPEG, max 10 MB.</p></div>")
        file_in = gr.File(label="Upload X-Ray", file_types=[".png", ".jpg", ".jpeg"],
                          type="filepath", height=240)
        preview = gr.Image(label="Preview", visible=False, interactive=False, height=260)
        status = gr.HTML(visible=False)
        with gr.Row():
            back_btn = gr.Button("← Back", elem_classes="btn-dark")
            cont_btn = gr.Button("Continue →", elem_classes="btn-primary", interactive=False)
        gr.HTML(f"<div class='disclaimer'>{DISCLAIMER}</div>")

    # ===== PAGE 3: Report =====
    with gr.Column(visible=False) as report:
        gr.HTML("<div class='page-title'><h1>Medical Analysis Report</h1></div>")
        gr.HTML("<div class='success'>✅ Report generated successfully.</div>")
        pdf_view = gr.HTML()
        with gr.Row():
            view_btn = gr.Button("View Report", elem_classes="btn-dark")
            dl_btn = gr.DownloadButton("Download PDF")
            new_btn = gr.Button("New Analysis", elem_classes="btn-primary")
        gr.HTML(f"<div class='disclaimer'>{DISCLAIMER}</div>")

    pages = [landing, upload, report]

    start_btn.click(lambda: show("upload"), outputs=pages)
    back_btn.click(lambda: show("landing"), outputs=pages)
    file_in.change(on_file, inputs=file_in, outputs=[preview, status, cont_btn])
    cont_btn.click(run_analysis, inputs=[p_name, p_id, file_in],
                   outputs=[*pages, pdf_view, dl_btn, status])
    view_btn.click(lambda: None, js="() => window.scrollTo({top: 0, behavior: 'smooth'})")
    new_btn.click(lambda: show("upload"), outputs=pages).then(
        clear_all, outputs=[file_in, p_name, p_id, preview, status, cont_btn])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860,
                        allowed_paths=[OUTPUT_DIR], **_LAUNCH_KW)