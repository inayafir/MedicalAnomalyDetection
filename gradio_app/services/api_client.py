"""
Thin client for the existing FastAPI backend.

This module intentionally contains NO model-loading, inference, or
classification logic of its own. Every real result (classification,
confidence, bounding boxes, Grad-CAM, PDF report) comes from the backend's
actual REST API, which in turn runs the real trained ResNet-50 and YOLOv8m
checkpoints. This file only formats HTTP requests/responses.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("GRADIO_API_KEY", "")  # only needed if backend enforces one
REQUEST_TIMEOUT = 60  # seconds; prediction calls run real CNN/detector inference


class BackendError(Exception):
    """Raised for any backend call failure, with a human-readable message.
    Never exposes raw stack traces or internal paths to the UI layer."""


def _headers(extra: dict | None = None) -> dict:
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    if extra:
        headers.update(extra)
    return headers


def _request(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{BACKEND_URL}{path}"
    try:
        resp = requests.request(
            method, url, timeout=REQUEST_TIMEOUT, headers=_headers(kwargs.pop("headers", None)), **kwargs
        )
    except requests.exceptions.ConnectionError as e:
        raise BackendError(
            f"Cannot reach the backend at {BACKEND_URL}. "
            "Is the FastAPI server running? (uvicorn app.main:app)"
        ) from e
    except requests.exceptions.Timeout as e:
        raise BackendError(
            "The backend took too long to respond. If this happened during "
            "analysis, the models may still be warming up — try again."
        ) from e

    if resp.status_code >= 400:
        detail = _extract_detail(resp)
        raise BackendError(f"{resp.status_code}: {detail}")

    return resp


def _extract_detail(resp: requests.Response) -> str:
    try:
        body = resp.json()
        detail = body.get("detail", resp.text)
        if isinstance(detail, list):  # FastAPI 422 validation errors
            return "; ".join(str(d.get("msg", d)) for d in detail)
        return str(detail)
    except ValueError:
        return resp.text[:300]


def check_health() -> dict:
    """Returns {'reachable': bool, 'status': str, 'model_loaded': bool}."""
    try:
        resp = _request("GET", "/health")
        body = resp.json()
        return {
            "reachable": True,
            "status": body.get("status", "unknown"),
            "model_loaded": bool(body.get("model_loaded", False)),
        }
    except BackendError:
        return {"reachable": False, "status": "unreachable", "model_loaded": False}


def create_patient(display_name: str) -> int:
    resp = _request("POST", "/patients", json={"display_name": display_name})
    return resp.json()["id"]


def upload_image(file_path: str, patient_id: int | None) -> dict:
    params = {}
    if patient_id is not None:
        params["patient_id"] = patient_id
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, _guess_content_type(file_path))}
        resp = _request("POST", "/images/upload", params=params, files=files)
    return resp.json()


def _guess_content_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "application/octet-stream")


def create_prediction(image_id: int) -> dict:
    resp = _request("POST", f"/predictions/{image_id}")
    return resp.json()


def create_report(prediction_id: int) -> dict:
    resp = _request("POST", f"/reports/{prediction_id}")
    return resp.json()


def fetch_file_bytes(relative_path: str) -> bytes:
    resp = _request("GET", f"/files/{relative_path}")
    return resp.content


@dataclass
class AnalysisResult:
    prediction_id: int
    image_id: int
    predicted_class: str
    confidence: float
    bboxes: list[dict]
    heatmap_path: str | None
