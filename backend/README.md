# Chest X-Ray Anomaly Detection — Backend

Backend REST API and database layer for a chest X-ray anomaly detection system, integrating with Person A's ML models (ResNet-50 classifier + YOLOv8m detector + Grad-CAM).

## Integration Notes

**Person A's repo:** `https://github.com/inayafir/MedicalAnomalyDetection`

### What was found

Two separate models exist — no single `predict()` function:

| Model | Entry Point | Returns |
|---|---|---|
| YOLOv8m detection | `src/detection/predict.py :: predict_image()` | `{"model", "detections": [...], "num_detections"}` |
| ResNet-50 + Grad-CAM | `src/classification/gradcam_resnet50.py :: generate_gradcam()` | `{"class", "confidence", "heatmap"}` |

### Classes (15 total, matching full VinBigData dataset)

| ID | Class Name | Type |
|---|---|---|
| 0 | Normal | Classification only (no bboxes) |
| 1 | Aortic enlargement | Detection + Classification |
| 2 | Atelectasis | Detection + Classification |
| 3 | Calcification | Detection + Classification |
| 4 | Cardiomegaly | Detection + Classification |
| 5 | Consolidation | Detection + Classification |
| 6 | ILD | Detection + Classification |
| 7 | Infiltration | Detection + Classification |
| 8 | Lung Opacity | Detection + Classification |
| 9 | Nodule/Mass | Detection + Classification |
| 10 | Other lesion | Detection + Classification |
| 11 | Pleural effusion | Detection + Classification |
| 12 | Pleural thickening | Detection + Classification |
| 13 | Pneumothorax | Detection + Classification |
| 14 | Pulmonary fibrosis | Detection + Classification |

### How integration works

`app/ml_interface/__init__.py` combines both models:
1. ResNet-50 provides the overall classification (15 classes)
2. YOLOv8m provides bounding box detections (14 disease classes)
3. Grad-CAM generates the heatmap overlay
4. All output is normalized to the shared contract shape

**No trained weights exist in the repo yet.** The system falls back to `ml_mock.py` automatically when checkpoints aren't found. 

> **Note:** The `.pt` and `.pth` weight file paths referenced below are **mock placeholders** — they do not contain real trained model weights. Once Person A completes training, real weight files should be placed at these exact paths to enable the actual ML inference pipeline.

When real weights are available, place them at:
- YOLO: `ml_core/runs/detect/outputs/detection/yolov8m_detection/weights/best.pt`
- ResNet-50: `ml_core/outputs/classification/resnet50/best_model.pth`

## Setup

```bash
cd backend
pip install -r requirements.txt
# Optionally clone Person A's repo into ml_core/:
#   git clone https://github.com/inayafir/MedicalAnomalyDetection ml_core
make run        # starts uvicorn with reload on http://localhost:8000
```

Interactive API docs: http://localhost:8000/docs

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./storage/dev.db` | SQLAlchemy database URL |
| `STORAGE_ROOT` | `./storage` | Root dir for uploaded images, heatmaps, reports |
| `MAX_UPLOAD_SIZE_MB` | `10` | Max upload size in MB |
| `ALLOWED_CONTENT_TYPES` | `image/png,image/jpeg` | Comma-separated allowed MIME types |
| `ML_DEVICE` | `cpu` | Torch device (`cpu` or `cuda`) |

## Prediction JSON Contract

```json
{
  "class": "Cardiomegaly",
  "confidence": 0.91,
  "bboxes": [
    {"class": "Cardiomegaly", "x1": 120, "y1": 80, "x2": 340, "y2": 260, "confidence": 0.87}
  ],
  "heatmap_path": "heatmaps/2026/08/25/3f2a1c.png"
}
```

**Valid classes:** All 15 VinBigData classes listed above.

## Database Schema

| Table | Key Columns | Notes |
|---|---|---|
| `patients` | `id`, `display_name`, `created_at` | Optional patient grouping |
| `images` | `id`, `patient_id` (FK), `file_path`, `original_filename`, `content_type`, `file_size_bytes`, `uploaded_at` | Uploaded X-ray images |
| `predictions` | `id`, `image_id` (FK), `predicted_class`, `confidence`, `bboxes` (JSON), `heatmap_path`, `created_at` | ML predictions |
| `reports` | `id`, `prediction_id` (FK), `pdf_path`, `generated_at` | Placeholder reports |

**Cascade:** Deleting an `Image` deletes its `Prediction` rows, which deletes their `Report` rows.

## Running Tests

```bash
make test
# or
pytest -v

# Skip slow integration tests:
pytest -m "not integration"

# Run only integration tests:
pytest -m "integration"
```

## Known Latency

On CPU, a single prediction takes approximately **2-5 seconds** (ResNet-50 inference + Grad-CAM generation). Plan frontend loading states accordingly.

## Known Gaps

- No authentication/authorization layer
- No PDF generation (`/reports` creates a DB row with `pdf_path = null`)
- No frontend/UI
- No deployment/CI configuration
- Trained model weights not yet available — system runs on mock fallback
