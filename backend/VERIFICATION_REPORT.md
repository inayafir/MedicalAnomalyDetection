# VERIFICATION REPORT

**Date:** 2026-08-31
**Backend:** Chest X-Ray Anomaly Detection API
**Auditor:** opencode (automated)
**Environment:** Python 3.8.10, Windows, SQLite

---

## 1. Requirements Check

### Models and contract

- ✅ **Real ResNet-50 checkpoint loads** — verified via `torch.load()`: checkpoint contains `class_names` with 5 classes matching `CLASSIFIER_CLASSES` exactly.
- ✅ **Real YOLO checkpoint loads** — verified via `YOLO()`: model has 14 classes matching `DETECTOR_CLASSES` exactly. (Note: prompt mentioned 15; actual checkpoint has 14. Code correctly uses 14.)
- ✅ **No mock references in request path** — `grep` for `ml_mock|mock_predict` in `app/` returns only the `ml_mock.py` file itself, which is never imported by any production code.
- ✅ **Top-level `class` from classifier, `bboxes[].class` from detector** — live test confirmed: `predicted_class=Pleural effusion` (in classifier set), `bboxes[].class` values are detector classes. Both sets are intentionally disjoint.
- ✅ **Bbox coordinates in original pixel space** — live test on 512×512 image confirmed all bbox coords within [0, 512] range.
- ✅ **Heatmap is real generated file** — served via `/files/`, 107KB PNG, verified as valid image via PIL.

### Endpoints (all 11 exist, documented, typed)

- ✅ `POST /images/upload` — returns 201 with `ImageResponse`
- ✅ `GET /images` — paginated, returns `PaginatedResponse[ImageListItem]`
- ✅ `GET /images/{id}` — returns `ImageDetail` with embedded `latest_prediction`
- ✅ `DELETE /images/{id}` — returns 204, cascades to predictions/reports/files
- ✅ `POST /predictions/{image_id}` — returns 201 with `PredictionRecord`
- ✅ `GET /predictions` — paginated, returns `PaginatedResponse[PredictionListItem]`
- ✅ `GET /predictions/{id}` — returns `PredictionRecord`
- ✅ `POST /reports/{prediction_id}` — returns 201 with `ReportResponse`
- ✅ `GET /patients` — paginated, returns `PaginatedResponse[PatientResponse]`
- ✅ `POST /patients` — returns 201 with `PatientResponse`
- ✅ `GET /files/{path}` — serves file with correct content-type
- ✅ `GET /health` — returns `HealthResponse` with `model_loaded` + `db_ok`

### Data layer

- ✅ **All 4 tables exist** — `patients`, `images`, `predictions`, `reports` with correct columns verified via SQLAlchemy model inspection.
- ✅ **FK constraints enforced** — `PRAGMA foreign_keys=ON` now enabled via event listener (was previously disabled, now fixed).
- ✅ **Cascade delete works** — ORM-level `cascade="all, delete-orphan"` + DB-level `ON DELETE CASCADE` both confirmed: deleting an image removes its predictions, reports, and filesystem files.
- ✅ **4 indexes present** — `ix_images_patient_id`, `ix_predictions_image_id`, `ix_predictions_created_at`, `ix_predictions_predicted_class`.

### Cross-cutting features

- ✅ **CORS configurable** — via `CORS_ORIGINS` env var, middleware present in `main.py`.
- ✅ **Pagination envelope consistent** — both `/images` and `/predictions` return `{items, total, limit, offset}`.
- ✅ **Path-traversal protection** — `safe_resolve()` in `storage.py` blocks `../../etc/passwd` (returns None → 404).
- ✅ **Consistent error shape** — all failures return `{"detail": "..."}` with correct status codes (400/404/413/415/422/500/503).

---

## 2. Functionality Check

### 1. Clean environment test

- ✅ `make run` starts server successfully
- ✅ `/health` returns `{"status": "ok", "model_loaded": true, "db_ok": true}` after ~8s startup
- ✅ No manual intervention required

### 2. Full happy-path flow

| Step | Status | Evidence |
|------|--------|----------|
| Upload image | ✅ 201 | `id=3, file_path=images/2026/08/31/...png` |
| GET image (no prediction) | ✅ 200 | `latest_prediction: null` |
| POST prediction | ✅ 201 | `predicted_class=Pleural effusion, confidence=0.71, elapsed=1.28s` |
| GET image (with prediction) | ✅ 200 | `latest_prediction` populated |
| GET heatmap | ✅ 200 | Valid PNG, 107KB, opens correctly |
| POST report | ✅ 201 | `id=1, prediction_id=8` |
| DELETE image | ✅ 204 | Image, predictions, reports, files all removed |

### 3. Failure-path spot checks

| Test | Status | Evidence |
|------|--------|----------|
| Wrong content-type | ✅ 415 | `"Content type 'text/plain' not allowed"` |
| Oversized upload | ✅ 413 | (tested via unit tests) |
| Corrupted image | ✅ 422 | (tested via unit tests) |
| Nonexistent image_id | ✅ 404 | `"Image with id 99999 not found"` |
| Nonexistent prediction_id | ✅ 404 | `"Prediction with id 99999 not found"` |
| Path traversal | ✅ 404 | `safe_resolve()` returns None |

### 4. Full test suite

- ✅ **54/54 tests pass** (52 fast + 2 integration), 0 failures
- Fast tests: ~69s
- Integration tests: ~14s (real model inference)

### 5. Performance sanity check

| Metric | Result |
|--------|--------|
| Avg prediction time | 1.85s |
| Max prediction time | 2.20s |
| Model reload per-request | No — models loaded once at startup |
| Concurrent (3 threads) | All succeed, max 3.67s |

### 6. Restart resilience

- ✅ SQLite file persists in `storage/dev.db`
- ✅ Previously uploaded images/predictions survive restart
- ✅ Models reload cleanly on startup

---

## 3. Integration Readiness

- ✅ **Docs render** — `/docs` and `/redoc` load without errors
- ✅ **OpenAPI schema complete** — all endpoints have typed request/response schemas, no bare `dict`/`Any`
- ✅ **Seed script works** — `python scripts/seed_demo_data.py` creates 3 patients, 3 images, 3 real predictions, 1 report. API is browsable immediately after.
- ✅ **CORS works** — `Origin: http://localhost:7860` → response includes `Access-Control-Allow-Origin: http://localhost:7860`
- ✅ **README complete** — documents all endpoints, two-label-set design, checkpoint config, curl examples, data model, known limitations
- ✅ **No secrets in `.env.example`** — contains only placeholder values
- ✅ **No stray `print()` in production code** — grep confirms zero hits in `app/`
- ✅ **No TODO/FIXME/HACK comments** — grep confirms zero hits in `app/`
- ⚠️ **Git not initialized** — no `.git` repo exists in the project directory. Git hygiene checks are N/A.
- ✅ **`.gitignore` covers** — `storage/`, `*.db`, `__pycache__/`, `.env`, `ml_core/checkpoints/*.pth`, `ml_core/checkpoints/*.pt`

---

## Issues Found and Fixed During This Audit

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | Seed script failed predictions (503) — `TestClient` doesn't run lifespan, so models weren't loaded | High | Added `load_models()` + `Base.metadata.create_all()` to seed script |
| 2 | Missing indexes on `predictions.image_id`, `predictions.predicted_class`, `images.patient_id` | Medium | Added `index=True` to model columns |
| 3 | SQLite `PRAGMA foreign_keys` not enabled — DB-level FK constraints silently ignored | High | Added `@event.listens_for(engine, "connect")` to set PRAGMA |
| 4 | `requirements.txt` missing ML dependencies (torch, torchvision, ultralytics, opencv-python) | Medium | Added to requirements.txt |
| 5 | `.gitignore` missing checkpoint files (334MB total) | Medium | Added `ml_core/checkpoints/*.pth`, `*.pt` |

---

## Blockers Before Push

**None.** All issues found during audit have been fixed and verified.

---

## Non-blocking Notes

1. **Git not initialized** — no `.git` directory exists. The project needs `git init` before pushing. This is not a code issue, just a setup step.
2. **YOLO has 14 classes, not 15** — the prompt requested "15 output classes" but the actual trained checkpoint has 14 disease classes (no "Normal" class). The code correctly uses 14. The README already documents this correctly.
3. **`ml_mock.py` still exists** — kept in the repo for reference/tests as specified. It is never imported by production code.
4. **Heatmap generation can fail silently** — `_generate_gradcam` catches all exceptions and returns `None`. The prediction still succeeds without a heatmap. This is intentional (degradation, not failure).
5. **Seed predictions all return "Pleural effusion"** — synthetic uniform-color images don't have meaningful features, so the classifier consistently predicts the same class. This is expected behavior for demo data.
