# MedicalAnomalyDetection

Functional prototype for automated anomaly detection in chest X-ray images using a hybrid ResNet-50 classifier and YOLOv8 detector with Grad-CAM explainability.

## Project Structure

```
MedicalAnomalyDetection/
  src/
    classification/     # ResNet-50 training, evaluation, Grad-CAM
    detection/          # YOLOv8 training, evaluation, prediction
    preprocessing/      # Data prep, WBF fusion, augmentation
  backend/              # FastAPI REST API + database (Person B)
    app/                # API code, DB models, ML integration
    tests/              # 27 tests (unit + integration)
    ml_core/            # (placeholder) ML repo clone goes here
```

## ML Models

- **ResNet-50** — 5-class chest X-ray classification (Normal, Cardiomegaly, Pleural effusion, Lung Opacity, Pulmonary fibrosis)
- **YOLOv8n** — Object detection with bounding boxes for abnormal findings
- **Grad-CAM** — Visual explanations overlaid on original X-rays

## Backend API

See `backend/README.md` for full documentation.

```bash
cd backend
pip install -r requirements.txt
make run    # http://localhost:8000/docs
```

> **Note:** The `.pt` and `.pth` weight file paths referenced in the codebase are **mock placeholders** — they do not contain real trained model weights. The backend currently runs on a mock prediction fallback. Once training is complete, real weight files should be placed at the documented paths.
