from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from app.config import settings
from app.ml_mock import mock_predict
from app.models import FindingClass
from app.storage import save_heatmap

# ---------------------------------------------------------------------------
# Integration notes (Person A's repo analysis):
#
# Two separate models exist:
#   1. YOLOv8m detection  -> src/detection/predict.py :: predict_image()
#      Returns: {"model": "YOLOv8m", "detections": [...], "num_detections": N}
#      Each detection: {"class_id", "class", "confidence", "bbox": {"x1","y1","x2","y2"}}
#      Bbox coords: original image pixels (via box.xyxy)
#      14 detection classes (all disease classes, Normal excluded)
#
#   2. ResNet-50 + Grad-CAM -> src/classification/gradcam_resnet50.py :: generate_gradcam()
#      Returns: {"class": str, "confidence": float, "heatmap": str (file path)}
#      15 classes (all VinBigData classes including Normal)
#
# Class label mapping (Person A -> contract):
#   "Normal"               -> "Normal"
#   "Aortic enlargement"   -> "Aortic enlargement"
#   "Atelectasis"          -> "Atelectasis"
#   "Calcification"        -> "Calcification"
#   "Cardiomegaly"         -> "Cardiomegaly"
#   "Consolidation"        -> "Consolidation"
#   "ILD"                  -> "ILD"
#   "Infiltration"         -> "Infiltration"
#   "Lung Opacity"         -> "Lung Opacity"
#   "Nodule/Mass"          -> "Nodule/Mass"
#   "Other lesion"         -> "Other lesion"
#   "Pleural effusion"     -> "Pleural effusion"
#   "Pleural thickening"   -> "Pleural thickening"
#   "Pneumothorax"         -> "Pneumothorax"
#   "Pulmonary fibrosis"   -> "Pulmonary fibrosis"
# ---------------------------------------------------------------------------

_MODEL_LOADED = False


def load_models():
    """Load Person A's models at startup. Called once from main.py."""
    global _MODEL_LOADED
    try:
        _try_load_yolo()
        _try_load_resnet()
        _MODEL_LOADED = True
    except Exception:
        _MODEL_LOADED = False


def _try_load_yolo():
    """Attempt to load YOLOv8 weights. Falls back gracefully if not found."""
    try:
        from ultralytics import YOLO

        model_path = (
            Path(__file__).resolve().parents[1]
            / "ml_core"
            / "runs"
            / "detect"
            / "outputs"
            / "detection"
            / "yolov8m_detection"
            / "weights"
            / "best.pt"
        )
        if model_path.exists():
            from app.ml_interface import _yolo as yolo_mod
            yolo_mod.model = YOLO(str(model_path))
        else:
            print(f"[ml_interface] YOLO weights not found at {model_path}, using mock")
    except ImportError:
        print("[ml_interface] ultralytics not installed, YOLO detection will use mock")


def _try_load_resnet():
    """Attempt to load ResNet-50 weights. Falls back gracefully if not found."""
    try:
        import torch
        import torch.nn as nn
        from torchvision import models

        model_path = (
            Path(__file__).resolve().parents[1]
            / "ml_core"
            / "outputs"
            / "classification"
            / "resnet50"
            / "best_model.pth"
        )
        if model_path.exists():
            device = torch.device(settings.ML_DEVICE)
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 15)
            checkpoint = torch.load(str(model_path), map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()

            from app.ml_interface import _resnet as resnet_mod
            resnet_mod.model = model
            resnet_mod.device = device
        else:
            print(f"[ml_interface] ResNet-50 weights not found at {model_path}, using mock")
    except ImportError:
        print("[ml_interface] torch not installed, classification will use mock")


def is_model_loaded() -> bool:
    return _MODEL_LOADED


def predict(image_path: str, original_width: int, original_height: int) -> dict:
    """
    Run prediction on an image.

    Attempts real models (YOLO + ResNet-50) if loaded, otherwise falls back to mock.
    Always returns the normalized contract shape.
    """
    result = _predict_real(image_path, original_width, original_height)
    if result is not None:
        return result

    return mock_predict(image_path, original_width, original_height)


def _predict_real(image_path: str, original_width: int, original_height: int) -> dict | None:
    """Try real models. Returns None if models aren't loaded."""
    abs_path = os.path.join(settings.STORAGE_ROOT, image_path)

    # --- Classification (ResNet-50) ---
    classification_result = _predict_resnet(abs_path)
    if classification_result is None:
        return None

    # --- Detection (YOLO) ---
    detections = _predict_yolo(abs_path, original_width, original_height)

    # --- Normalize bboxes to contract format ---
    bboxes = []
    for det in detections:
        bbox = det.get("bbox", {})
        bboxes.append({
            "class": det.get("class", classification_result["class"]),
            "x1": int(round(bbox.get("x1", 0))),
            "y1": int(round(bbox.get("y1", 0))),
            "x2": int(round(bbox.get("x2", 0))),
            "y2": int(round(bbox.get("y2", 0))),
            "confidence": det.get("confidence", 0.0),
        })

    # --- Generate Grad-CAM heatmap ---
    heatmap_path = _generate_gradcam(abs_path, classification_result["class_id"])

    return {
        "class": classification_result["class"],
        "confidence": classification_result["confidence"],
        "bboxes": bboxes,
        "heatmap_path": heatmap_path,
    }


def _predict_resnet(abs_path: str) -> dict | None:
    """Run ResNet-50 classification. Returns None if model not loaded."""
    try:
        from app.ml_interface import _resnet as resnet_mod
        if resnet_mod.model is None:
            return None

        import torch
        from torchvision import transforms

        IMAGE_SIZE = 224
        CLASS_NAMES = [
            "Normal", "Aortic enlargement", "Atelectasis", "Calcification",
            "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
            "Lung Opacity", "Nodule/Mass", "Other lesion",
            "Pleural effusion", "Pleural thickening", "Pneumothorax",
            "Pulmonary fibrosis",
        ]

        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = Image.open(abs_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(resnet_mod.device)

        with torch.no_grad():
            output = resnet_mod.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            class_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, class_id].item()

        return {
            "class": CLASS_NAMES[class_id],
            "class_id": class_id,
            "confidence": confidence,
        }
    except Exception:
        return None


def _predict_yolo(abs_path: str, original_width: int, original_height: int) -> list:
    """Run YOLO detection. Returns list of detection dicts."""
    try:
        from app.ml_interface import _yolo as yolo_mod
        if yolo_mod.model is None:
            return []

        results = yolo_mod.model(abs_path, conf=0.1, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                score = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                detections.append({
                    "class_id": class_id,
                    "class": class_name,
                    "confidence": round(score, 4),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                })
        return detections
    except Exception:
        return []


def _generate_gradcam(abs_path: str, target_class_id: int) -> str | None:
    """Generate Grad-CAM heatmap and save it. Returns path or None."""
    try:
        from app.ml_interface import _resnet as resnet_mod
        if resnet_mod.model is None:
            return None

        import cv2
        import numpy as np
        import torch
        from torchvision import transforms

        IMAGE_SIZE = 224

        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = Image.open(abs_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(resnet_mod.device)

        model = resnet_mod.model
        target_layer = model.layer4[-1].conv3
        activations = []
        gradients = []

        def fwd_hook(module, inp, out):
            activations.append(out)

        def bwd_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        model.zero_grad()
        output = model(input_tensor)
        score = output[0, target_class_id]
        score.backward()

        h1.remove()
        h2.remove()

        act = activations[0]
        grad = gradients[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam = cam.detach().cpu().numpy()

        original = cv2.imread(abs_path)
        h, w = original.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        _, buf = cv2.imencode(".png", overlay)
        heatmap_path = save_heatmap(buf.tobytes())
        return heatmap_path
    except Exception:
        return None
