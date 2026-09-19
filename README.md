# Medical X-Ray Anomaly Detection

Chest X-ray anomaly detection using a real trained **ResNet-50** classifier and
**YOLOv8m** detector, with **Grad-CAM** explainability, a **FastAPI** backend,
and two user interfaces: a **Gradio** app (the official demonstration
interface named in the project synopsis) and a full **React** frontend.

> **Academic/decision-support prototype.** Not a medical diagnosis. Results
> should be reviewed by a qualified medical professional.

## Project Structure

```
MedicalAnomalyDetection/
  src/                  # Training/evaluation code (classification, detection, preprocessing)
  backend/              # FastAPI REST API, database, real ML inference
    app/                # Routers, DB models, ML integration, PDF reporting
    tests/              # 74 tests (unit + integration)
    ml_core/checkpoints/ # Real trained weights go here (Git LFS)
  models/               # Same checkpoints, canonical location referenced by training code
  gradio_app/           # Gradio demonstration UI - official interface per synopsis
  xray-frontend/        # React frontend (alternative/extended UI)
  render.yaml           # Render Blueprint: web service + Postgres
```

## ML Models

- **ResNet-50** - 15-class chest X-ray classification (14 abnormality
  findings + Normal). Class names and order come directly from the trained
  checkpoint's embedded metadata, not hardcoded.
- **YOLOv8m** - 14-class abnormality detection with bounding boxes.
- **Grad-CAM** - Visual explanation heatmap overlaid on the original X-ray.

> The checkpoints are real trained artifacts, not stock pretrained weights -
> both carry embedded training metadata (epoch, validation metrics, class
> names, training args) confirming they came from this project's own
> training runs.

## Checkpoint Setup (required before running anything)

The real weight files are tracked via **Git LFS** and are NOT included in a
plain `git clone` without LFS, nor in the repository archive - you must pull
them explicitly:

```bash
git lfs install
git lfs pull
```

This should produce real binary files at:
- `backend/ml_core/checkpoints/resnet50.pth` (~270 MB)
- `backend/ml_core/checkpoints/yolov8m_14class.pt` (~50 MB)
- `models/resnet50/best_model.pth` (same file)
- `models/yolov8/best.pt` (same file)

**Verify you have the real files, not LFS pointers:**
```bash
ls -la backend/ml_core/checkpoints/
# Real files: tens/hundreds of MB. A ~130-byte file is an unresolved LFS pointer.
```

If a pointer slips through (e.g. GitHub LFS bandwidth quota exceeded), the
backend detects this at startup and refuses to load with a clear error
message - it will **never** silently fall back to a mock or pretrained model
for a medical inference path.

## Running the Backend

```bash
cd backend
pip install -r requirements-torch-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs: `http://127.0.0.1:8000/docs`

Installing torch via the CPU wheel index first (rather than plain `pip
install torch`) avoids pulling several GB of unused CUDA packages - this
matters both for local disk space and for keeping Docker/Render builds fast.

### Environment variables

See `backend/.env.example`. Key ones:
- `DATABASE_URL` - SQLite for local dev; Postgres required in production
- `ML_DEVICE` - `cpu` or `cuda`
- `RESNET_CHECKPOINT` / `YOLO_CHECKPOINT` - checkpoint paths
- `API_KEY`, `CORS_ORIGINS` - required (non-wildcard) in production

## Running the Gradio Interface (official demonstration UI)

The Gradio app is a thin client over the backend's real API - it contains no
model-loading or inference logic of its own; every classification,
detection, and heatmap shown comes from a real backend call.

```bash
# backend must already be running (see above)
cd gradio_app
pip install -r requirements.txt
python app.py
```

Opens at `http://127.0.0.1:7860`. Set `BACKEND_URL` env var if the backend
isn't at the default `http://127.0.0.1:8000`.

## Running the React Frontend

```bash
cd xray-frontend
npm install
npm run dev
```

See `xray-frontend/README.md` for details.

## PDF Reports

`POST /reports/{prediction_id}` generates a real PDF (via ReportLab)
containing patient info, the original X-ray, classification result,
detection table, and Grad-CAM heatmap - built entirely from actual
prediction data, never fabricated. Download via the returned `pdf_path`
through `GET /files/{pdf_path}`.

## Testing

```bash
cd backend
pytest
```

74 tests covering routes, database constraints, pagination, file handling,
CORS, production config validation, and report generation. (One additional
real-model integration test requires `ultralytics`/`torch` actually
installed with real weights present to run.)

## Docker / Deployment

See `render.yaml` for a one-click Render Blueprint (web service + managed
Postgres). Render resolves Git LFS content automatically during its clone
step. You must set `CORS_ORIGINS` to your real frontend URL after deploying
- the app refuses to start in production with a wildcard CORS origin.

## Known Limitations

- This is an academic prototype; performance metrics have not been
  independently clinically validated.
- CPU-only inference by design (no CUDA in the deployment target).
- Free-tier Render Postgres expires after 30 days with no backups.
- GitHub's free Git LFS tier has a monthly bandwidth quota; repeated deploys
  pulling ~320MB of weights can exceed it.

## Disclaimer

This system is an academic/decision-support prototype and is not a medical
diagnosis. Results should be reviewed by a qualified medical professional
before any clinical decision is made.
