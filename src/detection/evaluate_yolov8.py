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

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "detection"
    / "yolov8_detection"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "detection"
    / "yolov8_evaluation"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("YOLOv8 DETECTION EVALUATION")
    print("=" * 60)

    print(f"Data YAML : {DATA_YAML}")
    print(f"Model     : {MODEL_PATH}")
    print(f"Output    : {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found:\n{DATA_YAML}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found:\n{MODEL_PATH}\n\n"
            "Make sure YOLO training has produced best.pt."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading YOLOv8 model...")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nRunning validation...")

    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        project=str(OUTPUT_DIR),
        name="validation",
        plots=True,
        verbose=True,
    )

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    print("\n")
    print("=" * 60)
    print("YOLOv8 EVALUATION RESULTS")
    print("=" * 60)

    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"mAP@0.5        : {map50:.4f}")
    print(f"mAP@0.5:0.95   : {map50_95:.4f}")

    # --------------------------------------------------------
    # Per-class results
    # --------------------------------------------------------

    class_names = {
        0: "Cardiomegaly",
        1: "Pleural effusion",
        2: "Lung Opacity",
        3: "Pulmonary fibrosis",
    }

    print("\n")
    print("=" * 60)
    print("PER-CLASS RESULTS")
    print("=" * 60)

    # Ultralytics may return only classes that actually have
    # evaluation metrics. Therefore, never assume that the
    # metric arrays contain all 4 classes.

    p = metrics.box.p
    r = metrics.box.r
    ap50 = metrics.box.ap50
    ap = metrics.box.ap

    number_of_metric_classes = len(p)

    for class_id, class_name in class_names.items():

        if class_id < number_of_metric_classes:

            class_precision = float(p[class_id])
            class_recall = float(r[class_id])
            class_map50 = float(ap50[class_id])
            class_map50_95 = float(ap[class_id])

        else:

            class_precision = 0.0
            class_recall = 0.0
            class_map50 = 0.0
            class_map50_95 = 0.0

        print(
            f"{class_id}   "
            f"{class_name:<22} "
            f"P={class_precision:.4f} "
            f"R={class_recall:.4f} "
            f"mAP50={class_map50:.4f} "
            f"mAP50-95={class_map50_95:.4f}"
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = OUTPUT_DIR / "evaluation_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:

        f.write("YOLOv8 DETECTION EVALUATION\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Data: {DATA_YAML}\n\n")

        f.write("Overall Metrics\n")
        f.write("-" * 60 + "\n")

        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"mAP@0.5: {map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {map50_95:.4f}\n\n")

        f.write("Per-Class Metrics\n")
        f.write("-" * 60 + "\n")

        for class_id, class_name in class_names.items():

            if class_id < number_of_metric_classes:
                class_precision = float(p[class_id])
                class_recall = float(r[class_id])
                class_map50 = float(ap50[class_id])
                class_map50_95 = float(ap[class_id])
            else:
                class_precision = 0.0
                class_recall = 0.0
                class_map50 = 0.0
                class_map50_95 = 0.0

            f.write(
                f"{class_id} {class_name}: "
                f"P={class_precision:.4f}, "
                f"R={class_recall:.4f}, "
                f"mAP50={class_map50:.4f}, "
                f"mAP50-95={class_map50_95:.4f}\n"
            )

    print("\n")
    print("=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    print(f"Results directory : {OUTPUT_DIR}")
    print(f"Summary file      : {summary_path}")

    print("\nNext stage:")
    print("ResNet-50 classification pipeline")


if __name__ == "__main__":
    main()
    