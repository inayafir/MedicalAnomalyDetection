
from pathlib import Path
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------
# ResNet-50 model
# Final 15-class multi-label classification model
# ------------------------------------------------------------

RESNET_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet50"
    / "best_model.pth"
)


# ------------------------------------------------------------
# YOLOv8m model
# Final 14-class detection model
# ------------------------------------------------------------

YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolov8"
    / "best.pt"
)


# ------------------------------------------------------------
# Grad-CAM output
# Generated files remain inside outputs/
# ------------------------------------------------------------

GRADCAM_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "gradcam"
)

GRADCAM_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

# ResNet-50 has 15 classes.
# YOLOv8m has 14 abnormality classes and does not contain
# Normal because Normal images do not have bounding boxes.

NUM_CLASSES = 15

CLASS_NAMES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Normal",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]

IMAGE_SIZE = 224

# Threshold for ResNet multi-label classification.
CLASSIFICATION_THRESHOLD = 0.50

# YOLO confidence threshold.
YOLO_CONFIDENCE_THRESHOLD = 0.10

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    ),
])


# ============================================================
# LOAD RESNET-50
# ============================================================

def load_resnet50():

    print(
        "Loading ResNet-50..."
    )

    if not RESNET_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"ResNet-50 model not found:\n"
            f"{RESNET_MODEL_PATH}"
        )

    model = models.resnet50(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    checkpoint = torch.load(
        RESNET_MODEL_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "ResNet-50 loaded successfully."
    )

    return model


# ============================================================
# LOAD YOLOv8m
# ============================================================

def load_yolo():

    print(
        "Loading YOLOv8m..."
    )

    if not YOLO_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"YOLOv8m model not found:\n"
            f"{YOLO_MODEL_PATH}"
        )

    model = YOLO(
        str(YOLO_MODEL_PATH)
    )

    print(
        "YOLOv8m loaded successfully."
    )

    return model


# ============================================================
# RESNET-50 MULTI-LABEL PREDICTION
# ============================================================

def predict_resnet(
    model,
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = TRANSFORM(
        image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        # Multi-label classification:
        # each class has an independent probability.
        probabilities = torch.sigmoid(
            outputs
        )[0]

    classifications = []

    for class_id, probability in enumerate(
        probabilities
    ):

        confidence = float(
            probability.item()
        )

        if confidence >= CLASSIFICATION_THRESHOLD:

            classifications.append(
                {
                    "class_id": class_id,

                    "class": CLASS_NAMES[
                        class_id
                    ],

                    "confidence": round(
                        confidence,
                        4
                    )
                }
            )

    # Primary prediction:
    # class with the highest probability.
    predicted_class_id = int(
        torch.argmax(
            probabilities
        ).item()
    )

    primary_confidence = float(
        probabilities[
            predicted_class_id
        ].item()
    )

    return (
        predicted_class_id,
        primary_confidence,
        classifications
    )


# ============================================================
# YOLOv8m DETECTION
# ============================================================

def predict_yolo(
    model,
    image_path
):

    results = model(
        str(image_path),
        conf=YOLO_CONFIDENCE_THRESHOLD,
        verbose=False
    )

    detections = []

    for result in results:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = [
                float(value)
                for value in box.xyxy[0]
            ]

            class_name = result.names[
                class_id
            ]

            detections.append(
                {
                    "class_id": class_id,

                    "class": class_name,

                    "confidence": round(
                        confidence,
                        4
                    ),

                    "bbox": {
                        "x1": round(
                            x1,
                            2
                        ),

                        "y1": round(
                            y1,
                            2
                        ),

                        "x2": round(
                            x2,
                            2
                        ),

                        "y2": round(
                            y2,
                            2
                        )
                    }
                }
            )

    return detections


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.activations = None

        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )


    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output


    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]


    def generate(
        self,
        input_tensor,
        target_class
    ):

        self.model.zero_grad()

        output = self.model(
            input_tensor
        )

        score = output[
            0,
            target_class
        ]

        score.backward()

        gradients = self.gradients

        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = torch.relu(
            cam
        )

        cam = cam.squeeze()

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        return (
            cam
            .detach()
            .cpu()
            .numpy()
        )


    def close(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ============================================================
# CREATE GRAD-CAM
# ============================================================

def create_gradcam(
    model,
    image_path,
    class_id
):

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = TRANSFORM(
        image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        DEVICE
    )

    # Last convolutional layer of ResNet-50.
    target_layer = (
        model.layer4[-1].conv3
    )

    gradcam = GradCAM(
        model,
        target_layer
    )

    cam = gradcam.generate(
        input_tensor,
        class_id
    )

    gradcam.close()

    original = cv2.imread(
        str(image_path)
    )

    if original is None:

        raise FileNotFoundError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    height, width = (
        original.shape[:2]
    )

    cam = cv2.resize(
        cam,
        (width, height)
    )

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    output_path = (
        GRADCAM_OUTPUT_DIR
        / f"{image_path.stem}_gradcam.jpg"
    )

    cv2.imwrite(
        str(output_path),
        overlay
    )

    return output_path


# ============================================================
# COMBINED PREDICTION
# ============================================================

def predict(
    image_path,
    resnet_model,
    yolo_model
):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )


    # --------------------------------------------------------
    # ResNet multi-label classification
    # --------------------------------------------------------

    (
        class_id,
        confidence,
        classifications
    ) = predict_resnet(
        resnet_model,
        image_path
    )

    predicted_class = CLASS_NAMES[
        class_id
    ]


    # --------------------------------------------------------
    # YOLO detection
    # --------------------------------------------------------

    detections = predict_yolo(
        yolo_model,
        image_path
    )


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    heatmap_path = create_gradcam(
        resnet_model,
        image_path,
        class_id
    )


    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "image": str(
            image_path
        ),

        # Primary prediction
        "class": predicted_class,

        "confidence": round(
            confidence,
            4
        ),

        # All ResNet classes above threshold
        "classifications": classifications,

        # YOLO bounding boxes
        "bboxes": detections,

        "num_detections": len(
            detections
        ),

        # Grad-CAM heatmap
        "heatmap": str(
            heatmap_path
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "COMBINED ML PREDICTION PIPELINE"
    )

    print("=" * 60)

    print(
        f"Device : {DEVICE}"
    )

    print()

    # --------------------------------------------------------
    # Display model paths
    # --------------------------------------------------------

    print(
        f"ResNet-50 model:\n"
        f"{RESNET_MODEL_PATH}"
    )

    print()

    print(
        f"YOLOv8m model:\n"
        f"{YOLO_MODEL_PATH}"
    )

    print()

    # --------------------------------------------------------
    # Ask user for image
    # --------------------------------------------------------

    image_input = input(
        "Enter path to X-ray image: "
    ).strip()

    if not image_input:

        raise ValueError(
            "No image path was provided."
        )

    image_path = Path(
        image_input
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )

    print()

    print(
        f"Test image:\n"
        f"{image_path.resolve()}"
    )

    print()

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    resnet_model = load_resnet50()

    yolo_model = load_yolo()

    # --------------------------------------------------------
    # Run complete pipeline
    # --------------------------------------------------------

    result = predict(
        image_path,
        resnet_model,
        yolo_model
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL PREDICTION"
    )

    print(
        "=" * 60
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
