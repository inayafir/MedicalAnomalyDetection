from __future__ import annotations

import logging
import os
from pathlib import Path

from PIL import Image

from app.config import settings
from app.storage import save_heatmap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Two separate models with different class taxonomies:
#
#   ResNet-50 classifier → top-level `class` / `confidence` (5 classes)
#   YOLOv8m detector      → per-bbox `class` / `confidence` (14 disease classes)
#
# These are intentionally kept as separate label sets. The classifier gives
# a single whole-image label; the detector gives per-region boxes with their
# own finer-grained labels. Forcing them into one enum would silently lose
# information.
# ---------------------------------------------------------------------------

_MODEL_LOADED = False
_RESNET_CLASS_NAMES: list[str] = []
_YOLO_CLASS_NAMES: list[str] = []


def load_models():
    """Load both real checkpoints at startup. Fails loudly if either is missing or malformed."""
    global _MODEL_LOADED
    try:
        _load_yolo()
        _load_resnet()
        _warm_up()
        _MODEL_LOADED = True
        logger.info(
            "Models loaded: ResNet (%d classes), YOLO (%d classes)",
            len(_RESNET_CLASS_NAMES),
            len(_YOLO_CLASS_NAMES),
        )
    except Exception:
        logger.exception("Failed to load models — server will not serve predictions")
        _MODEL_LOADED = False


def _warm_up():
    """Run a small dummy inference to warm up CUDA/cuDNN."""
    try:
        import torch
        from app.ml_interface import _resnet as resnet_mod

        if resnet_mod.model is not None:
            dummy = torch.randn(1, 3, 224, 224).to(resnet_mod.device)
            with torch.no_grad():
                _ = resnet_mod.model(dummy)
            logger.info("ResNet-50 warm-up complete")
    except Exception:
        logger.debug("ResNet warm-up skipped")

    try:
        from app.ml_interface import _yolo as yolo_mod

        if yolo_mod.model is not None:
            logger.info("YOLOv8m warm-up: model ready")
    except Exception:
        pass


def _load_yolo():
    """Load YOLOv8 detector. Raises if checkpoint not found."""
    global _YOLO_CLASS_NAMES
    from ultralytics import YOLO

    model_path = Path(settings.YOLO_CHECKPOINT)
    if not model_path.is_absolute():
        # Resolve relative to backend/ root (two levels up from ml_interface/)
        model_path = Path(__file__).resolve().parents[2] / model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO checkpoint not found at: {model_path}\n"
            "This file is tracked via Git LFS. After cloning, run:\n"
            "  git lfs pull\n"
            "See README section 'Checkpoint setup' for details."
        )

    from app.ml_interface import _yolo as yolo_mod

    yolo_mod.model = YOLO(str(model_path))
    _YOLO_CLASS_NAMES = list(yolo_mod.model.names.values())

    if len(_YOLO_CLASS_NAMES) != 14:
        raise ValueError(
            f"Expected 14 detector classes, got {len(_YOLO_CLASS_NAMES)}: {_YOLO_CLASS_NAMES}"
        )

    logger.info("YOLOv8m loaded: %d classes from %s", len(_YOLO_CLASS_NAMES), model_path)


def _load_resnet():
    """Load ResNet-50 classifier. Raises if checkpoint not found or malformed."""
    global _RESNET_CLASS_NAMES
    import torch
    import torch.nn as nn
    from torchvision import models

    model_path = Path(settings.RESNET_CHECKPOINT)
    if not model_path.is_absolute():
        # Resolve relative to backend/ root (two levels up from ml_interface/)
        model_path = Path(__file__).resolve().parents[2] / model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"ResNet checkpoint not found at: {model_path}\n"
            "This file is tracked via Git LFS. After cloning, run:\n"
            "  git lfs pull\n"
            "See README section 'Checkpoint setup' for details."
        )

    device = torch.device(settings.ML_DEVICE)
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

    # Read class names from checkpoint
    if isinstance(checkpoint, dict) and "class_names" in checkpoint:
        _RESNET_CLASS_NAMES = list(checkpoint["class_names"])
    else:
        raise ValueError(
            "ResNet checkpoint missing 'class_names' key — cannot determine class order"
        )

    if len(_RESNET_CLASS_NAMES) != 5:
        raise ValueError(
            f"Expected 5 classifier classes, got {len(_RESNET_CLASS_NAMES)}: {_RESNET_CLASS_NAMES}"
        )

    num_classes = len(_RESNET_CLASS_NAMES)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    from app.ml_interface import _resnet as resnet_mod

    resnet_mod.model = model
    resnet_mod.device = device

    logger.info("ResNet-50 loaded: %d classes from %s", num_classes, model_path)


def is_model_loaded() -> bool:
    return _MODEL_LOADED


def get_classifier_classes() -> list[str]:
    """Return the 5 classifier class names (from ResNet checkpoint)."""
    return list(_RESNET_CLASS_NAMES)


def get_detector_classes() -> list[str]:
    """Return the 14 detector class names (from YOLO checkpoint)."""
    return list(_YOLO_CLASS_NAMES)


def predict(image_path: str, original_width: int, original_height: int) -> dict:
    """
    Run real inference on an image using both models.

    Returns the normalized contract dict. Raises if models aren't loaded.
    """
    if not _MODEL_LOADED:
        raise RuntimeError("Models not loaded — cannot run prediction")

    abs_path = os.path.join(settings.STORAGE_ROOT, image_path)

    # --- Classification (ResNet-50) → top-level class ---
    classification_result = _predict_resnet(abs_path)

    # --- Detection (YOLO) → per-bbox classes ---
    detections = _predict_yolo(abs_path, original_width, original_height)

    # --- Normalize bboxes ---
    bboxes = []
    for det in detections:
        bbox = det.get("bbox", {})
        bboxes.append({
            "class": det["class"],
            "x1": int(round(bbox.get("x1", 0))),
            "y1": int(round(bbox.get("y1", 0))),
            "x2": int(round(bbox.get("x2", 0))),
            "y2": int(round(bbox.get("y2", 0))),
            "confidence": det.get("confidence", 0.0),
        })

    # --- Grad-CAM heatmap ---
    heatmap_path = _generate_gradcam(abs_path, classification_result["class_id"])

    return {
        "class": classification_result["class"],
        "confidence": classification_result["confidence"],
        "bboxes": bboxes,
        "heatmap_path": heatmap_path,
    }


def _predict_resnet(abs_path: str) -> dict:
    """Run ResNet-50 classification. Raises on failure."""
    from app.ml_interface import _resnet as resnet_mod

    if resnet_mod.model is None or not _RESNET_CLASS_NAMES:
        raise RuntimeError("ResNet model not loaded")

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

    with torch.no_grad():
        output = resnet_mod.model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        class_id = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, class_id].item()

    return {
        "class": _RESNET_CLASS_NAMES[class_id],
        "class_id": class_id,
        "confidence": confidence,
    }


def _predict_yolo(abs_path: str, original_width: int, original_height: int) -> list:
    """Run YOLO detection. Raises on failure."""
    from app.ml_interface import _yolo as yolo_mod

    if yolo_mod.model is None:
        raise RuntimeError("YOLO model not loaded")

    # Ultralytics returns boxes in the input image's coordinate space.
    # When we pass the original file, boxes are in original pixel coords.
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
        logger.exception("Grad-CAM generation failed")
        return None
