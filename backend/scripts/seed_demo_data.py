"""
Seed demo data for the Chest X-Ray Anomaly Detection API.

Usage:
    cd backend
    python scripts/seed_demo_data.py

Creates sample patients, uploads sample X-ray images, and runs predictions
so Person C has non-empty data to build the UI against.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

# Add backend root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.ml_interface import load_models
from app.db import Base, engine


def create_sample_xray(color: tuple[int, int, int], size: tuple[int, int] = (512, 512)) -> bytes:
    """Create a synthetic X-ray-like image."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    print("=== Seeding demo data ===\n")
    Base.metadata.create_all(bind=engine)
    load_models()
    client = TestClient(app)

    # Create patients
    patients = []
    for name in ["John Doe", "Jane Smith", "Demo Patient"]:
        resp = client.post("/patients", json={"display_name": name})
        if resp.status_code == 201:
            patients.append(resp.json())
            print(f"  Created patient: {name} (id={resp.json()['id']})")
        else:
            print(f"  Patient '{name}' may already exist")

    # Upload sample images and run predictions
    samples = [
        ("chest_xray_1.png", (180, 180, 180)),
        ("chest_xray_2.png", (120, 120, 130)),
        ("chest_xray_3.png", (200, 200, 210)),
    ]

    for filename, color in samples:
        img_bytes = create_sample_xray(color)

        patient_id = patients[0]["id"] if patients else None
        url = "/images/upload"
        if patient_id:
            url += f"?patient_id={patient_id}"

        resp = client.post(
            url,
            files={"file": (filename, io.BytesIO(img_bytes), "image/png")},
        )
        if resp.status_code != 201:
            print(f"  Failed to upload {filename}: {resp.status_code} {resp.text}")
            continue

        image_id = resp.json()["id"]
        print(f"  Uploaded {filename} (image_id={image_id})")

        # Run prediction
        pred_resp = client.post(f"/predictions/{image_id}")
        if pred_resp.status_code == 201:
            pred = pred_resp.json()
            print(f"    Prediction: {pred['predicted_class']} ({pred['confidence']:.2f})")
        else:
            print(f"    Prediction failed: {pred_resp.status_code}")

    # Create a report
    resp = client.get("/predictions?limit=1")
    if resp.json()["items"]:
        pred_id = resp.json()["items"][0]["id"]
        report_resp = client.post(f"/reports/{pred_id}")
        if report_resp.status_code == 201:
            print(f"  Created report for prediction_id={pred_id}")

    # Summary
    images = client.get("/images")
    predictions = client.get("/predictions")
    print(f"\n=== Summary ===")
    print(f"  Patients:  {len(patients)}")
    print(f"  Images:    {images.json()['total']}")
    print(f"  Predictions: {predictions.json()['total']}")
    print("\nDemo data seeded successfully!")


if __name__ == "__main__":
    main()
