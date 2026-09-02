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

# YOLOv8m is the required detection model.
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

print(f"Device     : {DEVICE}")
print(f"Model      : {MODEL_NAME}")
print(f"Image size : {IMAGE_SIZE}")
print(f"Epochs     : {EPOCHS}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Data YAML  : {DATA_YAML}")


# ============================================================
# CHECK DATASET
# ============================================================

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"\nDataset YAML not found:\n{DATA_YAML}"
    )

print("\nDataset YAML found.")


# ============================================================
# LOAD PRETRAINED YOLOv8m
# ============================================================

print("\nLoading pretrained YOLOv8m model...")

model = YOLO(MODEL_NAME)

print("YOLOv8m model loaded successfully.")


# ============================================================
# TRAIN
# ============================================================

print("\nStarting YOLOv8m training...")

results = model.train(

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    data=str(DATA_YAML),

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device=DEVICE,

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    project=str(OUTPUT_DIR),
    name=PROJECT_NAME,

    # --------------------------------------------------------
    # Pretrained weights
    # --------------------------------------------------------
    pretrained=True,

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------
    patience=10,

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    workers=0,

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------
    save=True,
    save_period=-1,

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    val=True,
    split="val",

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------
    plots=True,

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------
    cache=False,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    seed=42,
    deterministic=True,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    verbose=True,
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("YOLOv8m TRAINING COMPLETED")
print("=" * 60)

RUN_DIR = OUTPUT_DIR / PROJECT_NAME

BEST_MODEL = RUN_DIR / "weights" / "best.pt"

print(f"\nOutput directory:")
print(RUN_DIR)

print("\nBest model:")
print(BEST_MODEL)

print("\nTraining results:")
print(RUN_DIR / "results.csv")

print("\nNext stage:")
print("Evaluate YOLOv8m detection model")