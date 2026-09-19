from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 15-class YOLO dataset
DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "detection"
    / "yolo"
    / "data.yaml"
)

# YOLOv8m evaluation output
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "detection"
    / "yolov8m_evaluation"
)

# YOLOv8m trained model
MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "detection"
    / "yolov8m_detection-6"
    / "weights"
    / "best.pt"
)


# ============================================================
# 15 CLASSES
# ============================================================

CLASS_NAMES = {
    0: "Aortic enlargement",
    1: "Atelectasis",
    2: "Calcification",
    3: "Cardiomegaly",
    4: "Consolidation",
    5: "ILD",
    6: "Infiltration",
    7: "Lung Opacity",
    8: "Nodule/Mass",
    9: "Other lesion",
    10: "Pleural effusion",
    11: "Pleural thickening",
    12: "Pneumothorax",
    13: "Pulmonary fibrosis",
    14: "No finding",
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("YOLOv8m 15-CLASS DETECTION EVALUATION")
    print("=" * 70)

    print(f"\nData YAML : {DATA_YAML}")
    print(f"Model     : {MODEL_PATH}")
    print(f"Output    : {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"\ndata.yaml not found:\n{DATA_YAML}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nYOLOv8m model not found:\n{MODEL_PATH}\n\n"
            "Make sure YOLOv8m training has produced best.pt."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING YOLOv8m MODEL")
    print("=" * 70)

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING VALIDATION")
    print("=" * 70)

    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        project=str(OUTPUT_DIR),
        name="validation",
        plots=True,
        verbose=True,
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    # Calculate F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # --------------------------------------------------------
    # Print overall results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("YOLOv8m EVALUATION RESULTS")
    print("=" * 70)

    print(f"Precision      : {precision:.4f}  ({precision * 100:.2f}%)")
    print(f"Recall         : {recall:.4f}  ({recall * 100:.2f}%)")
    print(f"F1 Score       : {f1:.4f}  ({f1 * 100:.2f}%)")
    print(f"mAP@0.5        : {map50:.4f}  ({map50 * 100:.2f}%)")
    print(f"mAP@0.5:0.95   : {map50_95:.4f}  ({map50_95 * 100:.2f}%)")

    # --------------------------------------------------------
    # Per-class results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PER-CLASS RESULTS")
    print("=" * 70)

    p = metrics.box.p
    r = metrics.box.r
    ap50 = metrics.box.ap50
    ap = metrics.box.ap

    number_of_metric_classes = len(p)

    print(
        f"{'ID':<5}"
        f"{'Class':<25}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'mAP50':<12}"
        f"{'mAP50-95':<12}"
    )

    print("-" * 78)

    per_class_results = []

    for class_id, class_name in CLASS_NAMES.items():

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
            f"{class_id:<5}"
            f"{class_name:<25}"
            f"{class_precision:<12.4f}"
            f"{class_recall:<12.4f}"
            f"{class_map50:<12.4f}"
            f"{class_map50_95:<12.4f}"
        )

        per_class_results.append({
            "id": class_id,
            "name": class_name,
            "precision": class_precision,
            "recall": class_recall,
            "map50": class_map50,
            "map50_95": class_map50_95,
        })

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = OUTPUT_DIR / "evaluation_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:

        f.write("YOLOv8m 15-CLASS DETECTION EVALUATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Data: {DATA_YAML}\n")
        f.write("Split: val\n")
        f.write("Image size: 416\n\n")

        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-" * 70 + "\n")

        f.write(
            f"Precision: {precision:.4f} "
            f"({precision * 100:.2f}%)\n"
        )

        f.write(
            f"Recall: {recall:.4f} "
            f"({recall * 100:.2f}%)\n"
        )

        f.write(
            f"F1 Score: {f1:.4f} "
            f"({f1 * 100:.2f}%)\n"
        )

        f.write(
            f"mAP@0.5: {map50:.4f} "
            f"({map50 * 100:.2f}%)\n"
        )

        f.write(
            f"mAP@0.5:0.95: {map50_95:.4f} "
            f"({map50_95 * 100:.2f}%)\n\n"
        )

        # Per-class metrics
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 70 + "\n")

        for result in per_class_results:

            f.write(
                f"{result['id']} "
                f"{result['name']}: "
                f"P={result['precision']:.4f}, "
                f"R={result['recall']:.4f}, "
                f"mAP50={result['map50']:.4f}, "
                f"mAP50-95={result['map50_95']:.4f}\n"
            )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nResults directory:")
    print(OUTPUT_DIR)

    print(f"\nSummary file:")
    print(summary_path)

    print("\nGenerated evaluation files may include:")
    print("- evaluation_summary.txt")
    print("- confusion_matrix.png")
    print("- confusion_matrix_normalized.png")
    print("- PR_curve.png")
    print("- F1_curve.png")
    print("- P_curve.png")
    print("- R_curve.png")

    print("\nNext stage:")
    print("ResNet-50 classification pipeline")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()