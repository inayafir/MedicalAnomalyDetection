# Lumen — Chest X-Ray Analysis Frontend

Person C's frontend for the MedicalAnomalyDetection project. A React + Vite +
Tailwind CSS single-page app that talks to Person B's existing FastAPI
backend over REST — no ML logic lives here.

## Setup

```bash
npm install
cp .env.example .env   # edit VITE_API_BASE_URL if the backend isn't on :8000
npm run dev
```

The backend must be running separately (`uvicorn app.main:app --reload` from
`backend/`) with CORS allowing this frontend's origin (`CORS_ORIGINS` in the
backend's `.env`).

## Pages

- `/` — Dashboard: total/normal/abnormal counts (from `GET /predictions`),
  recent analyses, backend + model status (`GET /health`).
- `/upload` — Drag-and-drop upload with PNG/JPEG + 10 MB validation, optional
  numeric patient ID, then automatically runs `POST /images/upload` then
  `POST /predictions/{image_id}`.
- `/studies` — Browsable grid of uploaded images (`GET /images`); opening one
  jumps to its latest prediction, or runs a new one if it doesn't have one
  yet.
- `/analysis/:predictionId` — Full results: original image, bounding-box
  overlay, Grad-CAM toggle, detection table, classification confidence,
  image metadata.
- `/report/:predictionId` — Generates a report record (`POST
  /reports/{prediction_id}`) and renders a printable preliminary report.
  The backend doesn't generate PDFs or an LLM summary yet (see Known gaps
  below), so the "summary" here is built directly, in the browser, from the
  structured prediction data, not a model-generated narrative, and
  "Download / print" uses the browser's native print-to-PDF.

## Important backend realities this frontend is built against

The docx synopsis and the actual repo code have drifted apart in a couple of
places, this frontend follows the code, not the docx:

- Classifier is 5-class, not 15: Normal, Cardiomegaly, Pleural effusion,
  Lung Opacity, Pulmonary fibrosis (backend/app/models.py,
  CLASSIFIER_CLASSES). The detector is still the 14-class disease-only
  YOLOv8m set.
- No mock fallback is wired up. POST /predictions/{image_id} returns 503 if
  is_model_loaded() is false, and right now the checkpoint files in
  ml_core/checkpoints/ are tiny placeholders (~130 bytes), so real weights
  aren't loaded. The Upload page shows a specific message for this 503 case
  instead of a generic error. ml_mock.py exists in the repo but isn't called
  from the predictions router in this version.
- Field names follow the real API, not the docx's normalized-contract
  naming: the top-level field is predicted_class (not class), and each bbox
  uses class/confidence/x1/y1/x2/y2.
- There's no GET /reports/{id}, only POST /reports/{prediction_id}, which
  creates a new row each call and returns it directly (pdf_path is always
  null). The Report page keeps the generated report in memory rather than
  trying to refetch it.

## What's intentionally not here

- No Ollama integration or backend PDF rendering, the current backend
  doesn't have either. If you want those, they'd need to be added to the
  FastAPI backend first (app/services/ollama_service.py, a PDF renderer),
  and this frontend can be pointed at whatever routes that produces.
- No authentication, matches the backend, which is fully open.

## Tech

React 19, Vite, Tailwind CSS v4 (via @tailwindcss/vite), React Router,
Axios, lucide-react/recharts installed but only pulled in where used.
