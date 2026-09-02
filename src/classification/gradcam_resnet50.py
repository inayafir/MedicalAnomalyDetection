from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "resnet50"
    / "best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "gradcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

NUM_CLASSES = 5

CLASS_NAMES = [
    "Normal",
    "Cardiomegaly",
    "Pleural effusion",
    "Lung Opacity",
    "Pulmonary fibrosis",
]

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"ResNet-50 weights not found:\n{MODEL_PATH}"
        )

    model = models.resnet50(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._save_activation
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._save_gradient
            )
        )


    def _save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output


    def _save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]


    def generate(
        self,
        input_tensor,
        target_class=None
    ):

        self.model.zero_grad()

        output = self.model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        if target_class is None:

            target_class = (
                torch.argmax(
                    probabilities,
                    dim=1
                ).item()
            )

        score = output[
            0,
            target_class
        ]

        score.backward()

        activations = self.activations
        gradients = self.gradients

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

        cam = cam.detach().cpu().numpy()

        return (
            cam,
            target_class,
            probabilities
        )


    def close(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap(
    image_path,
    cam
):

    original = cv2.imread(
        str(image_path)
    )

    if original is None:

        raise FileNotFoundError(
            f"Could not read image:\n{image_path}"
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

    return overlay


# ============================================================
# RUN GRAD-CAM
# ============================================================

def generate_gradcam(
    image_path,
    output_path=None
):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    model = load_model()

    # Final convolutional layer of ResNet-50
    target_layer = (
        model.layer4[-1].conv3
    )

    gradcam = GradCAM(
        model,
        target_layer
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = TRANSFORM(
        image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        DEVICE
    )

    cam, class_id, probabilities = (
        gradcam.generate(
            input_tensor
        )
    )

    gradcam.close()

    predicted_class = (
        CLASS_NAMES[class_id]
    )

    confidence = (
        probabilities[0, class_id]
        .item()
    )

    overlay = create_heatmap(
        image_path,
        cam
    )

    if output_path is None:

        output_path = (
            OUTPUT_DIR
            / f"{image_path.stem}_gradcam.jpg"
        )

    output_path = Path(
        output_path
    )

    cv2.imwrite(
        str(output_path),
        overlay
    )

    return {
        "class": predicted_class,
        "confidence": confidence,
        "heatmap": str(output_path),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RESNET-50 GRAD-CAM")
    print("=" * 60)

    print(
        f"Device : {DEVICE}"
    )

    print(
        f"Model  : {MODEL_PATH}"
    )

    image_input = input(
        "\nEnter path to X-ray image: "
    ).strip()

    result = generate_gradcam(
        image_input
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "GRAD-CAM RESULT"
    )

    print(
        "=" * 60
    )

    print(
        f"Class      : {result['class']}"
    )

    print(
        f"Confidence : {result['confidence']:.4f}"
    )

    print(
        f"Heatmap    : {result['heatmap']}"
    )