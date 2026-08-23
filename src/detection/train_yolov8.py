from pathlib import Path

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

MODEL_NAME = "yolov8n.pt"

IMAGE_SIZE = 640

EPOCHS = 10

BATCH_SIZE = 8

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

PROJECT_NAME = "yolov8_detection"


# ============================================================
# START
# ============================================================

print("=" * 60)
print("YOLOv8 DETECTION TRAINING")
print("=" * 60)

print(f"Device     : {DEVICE}")
print(f"Model      : {MODEL_NAME}")
print(f"Image size : {IMAGE_SIZE}")
print(f"Epochs     : {EPOCHS}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Data YAML  : {DATA_YAML}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading pretrained YOLOv8 model...")

model = YOLO(
    MODEL_NAME
)

print("YOLOv8 model loaded successfully.")


# ============================================================
# TRAIN
# ============================================================

print("\nStarting YOLOv8 training...")

results = model.train(
    data=str(DATA_YAML),

    epochs=EPOCHS,

    imgsz=IMAGE_SIZE,

    batch=BATCH_SIZE,

    device=DEVICE,

    project=str(OUTPUT_DIR),

    name=PROJECT_NAME,

    pretrained=True,

    patience=3,

    workers=0,

    verbose=True,

    plots=True,
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("YOLOv8 TRAINING COMPLETED")
print("=" * 60)

print(
    f"Output directory : "
    f"{OUTPUT_DIR / PROJECT_NAME}"
)

print("\nNext stage:")
print("Evaluate YOLOv8 detection model")
