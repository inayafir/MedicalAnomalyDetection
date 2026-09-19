# Chest X-Ray Anomaly Detection — Backend

Backend REST API and database layer for a chest X-ray anomaly detection system, integrating with Person A's ML models (ResNet-50 classifier + YOLOv8m detector + Grad-CAM).

## 1. Quickstart

### Prerequisites

- Python 3.8+
- **Git LFS** (required for checkpoint files — see below)

### Install and run

```bash
# If you just cloned this repo, pull the model checkpoints via Git LFS:
git lfs pull

cd backend
pip install -r requirements.txt

# Seed demo data (creates patients, uploads sample images, runs predictions)
make seed

# Start the server
make run
```

Server runs at **http://localhost:8000**. Interactive docs at **http://localhost:8000/docs**.

### Checkpoint setup

This project uses **Git LFS** to track the model checkpoint files (`resnet50.pth`, `yolov8m_14class.pt`). If you cloned without LFS installed, the checkpoint files will be text pointer stubs and the server will fail at startup with an actionable error message.

To fix: install [Git LFS](https://git-lfs.com/), then run `git lfs pull` from the repo root.

**Docker alternative:**
```bash
cd backend
docker compose up --build
```

## 2. Two Label Sets — Why They're Different

The backend uses **two separate models** that output **different label sets** for different tasks. They are intentionally not merged into one enum.

### Classifier (ResNet-50, 15 classes)

Produces a single **whole-image label**. This is the top-level `predicted_class` field.

| Index | Class Name |
|-------|------------|
| 0 | Aortic enlargement |
| 1 | Atelectasis |
| 2 | Calcification |
| 3 | Cardiomegaly |
| 4 | Consolidation |
| 5 | ILD |
| 6 | Infiltration |
| 7 | Lung Opacity |
| 8 | Nodule/Mass |
| 9 | Normal |
| 10 | Other lesion |
| 11 | Pleural effusion |
| 12 | Pleural thickening |
| 13 | Pneumothorax |
| 14 | Pulmonary fibrosis |

Source: `checkpoint["class_names"]` from `ml_core/checkpoints/resnet50.pth`

### Detector (YOLOv8m, 14 disease classes)

Produces **per-region bounding boxes**, each with its own label. These are the `bboxes[].class` fields. The detector has no "Normal" class — if nothing is detected, the bboxes list is empty.

| Index | Class Name |
|-------|------------|
| 0 | Aortic enlargement |
| 1 | Atelectasis |
| 2 | Calcification |
| 3 | Cardiomegaly |
| 4 | Consolidation |
| 5 | ILD |
| 6 | Infiltration |
| 7 | Lung Opacity |
| 8 | Nodule/Mass |
| 9 | Other lesion |
| 10 | Pleural effusion |
| 11 | Pleural thickening |
| 12 | Pneumothorax |
| 13 | Pulmonary fibrosis |

Source: `model.names` from `ml_core/checkpoints/yolov8m_14class.pt`

**Why two sets?** The classifier gives a single overall impression (one label for the whole image). The detector finds specific regions and labels them with finer granularity. For example, the classifier might say "Normal" while the detector still finds a small "Atelectasis" region — or the classifier might say "Cardiomegaly" while the detector also finds "Aortic enlargement" in the same image. Both pieces of information are useful; collapsing them into one set would silently lose information.

If the frontend needs a simplified "does this box match the overall finding" view, that's a **display decision** — not something baked into the backend contract.

### The 14 detector names are a subset of the 15 classifier names

The 14 YOLO classes are exactly the 14 VinBigData abnormality findings. The 15 ResNet classes are those same 14 findings plus `"Normal"` (i.e. "no finding detected"). `"Normal"` is the only class present in the classifier but absent from the detector, because you cannot draw a bounding box for the *absence* of a finding.

### Edge case: "Normal" classifier + non-empty bboxes

This is a valid scenario — the classifier's holistic assessment is "Normal" but YOLO still identifies small regions. The backend passes both outputs through without reconciliation; the frontend should decide how to display this (e.g. show a "Normal" badge but still render the bboxes). This is documented here so Person C's UI doesn't assume the two always agree.

## 3. API Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/images/upload` | POST | Upload an X-ray, get back an image record |
| `/images` | GET | Paginated list of uploaded images |
| `/images/{id}` | GET | One image + its latest prediction |
| `/images/{id}` | DELETE | Delete an image (cascades) |
| `/predictions/{image_id}` | POST | Run inference on an image, persist and return result |
| `/predictions` | GET | Paginated list of predictions |
| `/predictions/{id}` | GET | One prediction record |
| `/reports/{prediction_id}` | POST | Create a report placeholder row |
| `/patients` | GET/POST | Basic patient list/create |
| `/files/{path}` | GET | Serve a stored image/heatmap file by its relative path |
| `/health` | GET | Liveness + model/DB status |

### curl Examples

**Upload an image:**
```bash
curl -X POST http://localhost:8000/images/upload \
  -F "file=@chest_xray.png" \
  -F "patient_id=1"
```

**Run prediction:**
```bash
curl -X POST http://localhost:8000/predictions/1
```

**Example prediction response** (top-level `class` from 15-class classifier, bbox classes from 14-class detector):
```json
{
  "id": 1,
  "image_id": 1,
  "predicted_class": "Cardiomegaly",
  "confidence": 0.91,
  "bboxes": [
    {"class_": "Aortic enlargement", "confidence": 0.87, "x1": 120, "y1": 80, "x2": 340, "y2": 260},
    {"class_": "Pleural thickening", "confidence": 0.62, "x1": 40, "y1": 300, "x2": 190, "y2": 410}
  ],
  "heatmap_path": "heatmaps/2026/08/31/abc123.png",
  "created_at": "2026-08-31T21:00:00Z"
}
```

**Example showing legitimate model disagreement:**
```json
{
  "predicted_class": "Normal",
  "confidence": 0.60,
  "bboxes": [
    {"class_": "Atelectasis", "confidence": 0.75, "x1": 10, "y1": 20, "x2": 100, "y2": 200}
  ],
  "heatmap_path": "heatmaps/2026/08/31/def456.png"
}
```

**Other endpoints:**
```bash
# List images
curl "http://localhost:8000/images?limit=10&offset=0"

# Get image detail
curl http://localhost:8000/images/1

# Delete image
curl -X DELETE http://localhost:8000/images/1

# List predictions (filtered)
curl "http://localhost:8000/predictions?limit=10&predicted_class=Cardiomegaly"

# Get prediction
curl http://localhost:8000/predictions/1

# Create report
curl -X POST http://localhost:8000/reports/1

# Create patient
curl -X POST http://localhost:8000/patients \
  -H "Content-Type: application/json" \
  -d '{"display_name": "John Doe"}'

# List patients
curl http://localhost:8000/patients

# Serve file
curl http://localhost:8000/files/images/2026/08/31/abc123.png

# Health check
curl http://localhost:8000/health
```

## 4. Data Model

### Tables

| Table | Columns | Notes |
|---|---|---|
| `patients` | `id` (PK), `display_name`, `created_at` | Optional patient grouping |
| `images` | `id` (PK), `patient_id` (FK), `file_path`, `original_filename`, `content_type`, `file_size_bytes`, `uploaded_at` | Uploaded X-ray images |
| `predictions` | `id` (PK), `image_id` (FK), `predicted_class`, `confidence`, `bboxes` (JSON), `heatmap_path`, `created_at` | ML predictions |
| `reports` | `id` (PK), `prediction_id` (FK), `pdf_path`, `generated_at` | Placeholder reports |

### Cascade Delete

Deleting an `Image` cascades to delete its `Prediction` rows, which cascade to delete their `Report` rows. Filesystem files (image + heatmaps) are also removed.

## 5. Where Files Live

### Checkpoints

```
backend/ml_core/checkpoints/
├── resnet50.pth              # ResNet-50 classifier (15 classes)
└── yolov8m_14class.pt        # YOLOv8m detector (14 disease classes)
```

Paths are configurable via `RESNET_CHECKPOINT` and `YOLO_CHECKPOINT` env vars. To swap checkpoints, update the env vars — the server validates class counts at startup (15 for ResNet, 14 for YOLO) and fails loudly if they don't match.

### Storage

```
storage/
├── images/YYYY/MM/DD/{uuid}.png      # uploaded X-rays
├── heatmaps/YYYY/MM/DD/{uuid}.png    # Grad-CAM overlays
├── reports/{prediction_id}/{uuid}.pdf # report PDFs (future)
└── dev.db                            # SQLite database
```

To construct a URL for an image or heatmap from a prediction record:
```
http://localhost:8000/files/{heatmap_path}
```

## 6. Known Limitations

- **No authentication/authorization** — all endpoints are public
- **No PDF generation** — `/reports` creates a DB row with `pdf_path = null`
- **CORS wide open by default** (`*`) — lock down `CORS_ORIGINS` env var before deployment
- **Inference latency on CPU**: ~2-5 seconds per prediction (ResNet-50 + Grad-CAM). Plan frontend loading states accordingly.
- **Classifier and detector may disagree** — the backend passes both outputs through; the frontend should handle cases where e.g. classifier says "Normal" but detector still finds regions
- **No deployment/CI configuration**

## 7. Running Tests

```bash
# All tests (fast + integration)
make test

# Fast tests only (no real model inference, runs in ~5s)
make test-fast

# Integration tests only (requires model weights)
pytest -m "integration" -v
```

## 8. Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./storage/dev.db` | SQLAlchemy database URL |
| `STORAGE_ROOT` | `./storage` | Root dir for uploaded images, heatmaps, reports |
| `MAX_UPLOAD_SIZE_MB` | `10` | Max upload size in MB |
| `ALLOWED_CONTENT_TYPES` | `image/png,image/jpeg` | Comma-separated allowed MIME types |
| `ML_DEVICE` | `cpu` | Torch device (`cpu` or `cuda`) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins for CORS |
| `RESNET_CHECKPOINT` | `ml_core/checkpoints/resnet50.pth` | Path to ResNet-50 weights |
| `YOLO_CHECKPOINT` | `ml_core/checkpoints/yolov8m_14class.pt` | Path to YOLOv8m weights |
