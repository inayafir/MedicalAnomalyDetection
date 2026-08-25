from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "detection"
    / "yolo"
    / "data.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "detection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

# Project requirement:
# YOLOv8m detector
#
# Normal is NOT a detection class.
#
# Detection classes:
# 0 - Cardiomegaly
# 1 - Pleural effusion
# 2 - Lung Opacity
# 3 - Pulmonary fibrosis

MODEL_NAME = "yolov8m.pt"

IMAGE_SIZE = 416

EPOCHS = 10

BATCH_SIZE = 4

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

PROJECT_NAME = "yolov8m_detection"

RUN_NAME = "train"


# ============================================================
# START
# ============================================================

print("=" * 60)
print("YOLOv8m DETECTION TRAINING")
print("=" * 60)

print(
    f"Device     : {DEVICE}"
)

print(
    f"Model      : {MODEL_NAME}"
)

print(
    f"Image size : {IMAGE_SIZE}"
)

print(
    f"Epochs     : {EPOCHS}"
)

print(
    f"Batch size : {BATCH_SIZE}"
)

print(
    f"Data YAML  : {DATA_YAML}"
)


# ============================================================
# CHECK DATASET YAML
# ============================================================

if not DATA_YAML.exists():

    raise FileNotFoundError(
        f"\nDataset YAML not found:\n"
        f"{DATA_YAML}"
    )


print(
    "\nDataset YAML found."
)


# ============================================================
# LOAD PRETRAINED YOLOv8m
# ============================================================

print(
    "\nLoading pretrained YOLOv8m model..."
)

model = YOLO(
    MODEL_NAME
)

print(
    "YOLOv8m model loaded successfully."
)


# ============================================================
# TRAIN
# ============================================================

print(
    "\nStarting YOLOv8m training..."
)

results = model.train(

    # Dataset
    data=str(DATA_YAML),

    # Training duration
    epochs=EPOCHS,

    # Input resolution
    imgsz=IMAGE_SIZE,

    # Batch size
    batch=BATCH_SIZE,

    # CPU/GPU
    device=DEVICE,

    # Output
    project=str(OUTPUT_DIR),

    name=PROJECT_NAME,

    # Pretrained model
    pretrained=True,

    # Early stopping
    patience=10,

    # DataLoader
    workers=0,

    # Save checkpoints
    save=True,

    # Save best model
    save_period=-1,

    # Validation
    val=True,

    # Generate plots
    plots=True,

    # Cache disabled to avoid unnecessary RAM usage
    cache=False,

    # Reproducibility
    seed=42,

    # Verbose training output
    verbose=True,
)


# ============================================================
# COMPLETED
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "YOLOv8m TRAINING COMPLETED"
)

print(
    "=" * 60
)

print(
    f"Output directory:\n"
    f"{OUTPUT_DIR / PROJECT_NAME}"
)

print(
    "\nThe trained weights should be located at:"
)

print(
    OUTPUT_DIR
    / PROJECT_NAME
    / RUN_NAME
    / "weights"
    / "best.pt"
)

print(
    "\nNext stage:"
)

print(
    "Evaluate YOLOv8m detection model"
)