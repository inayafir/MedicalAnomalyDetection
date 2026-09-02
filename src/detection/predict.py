from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "detection"
    / "yolov8m_detection-6"
    / "weights"
    / "best.pt"
)


# ============================================================
# YOLO CLASS NAMES
# ============================================================

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
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "No finding",
]


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"YOLOv8m model not found:\n{MODEL_PATH}"
    )


print(
    f"Loading YOLOv8m model:\n{MODEL_PATH}"
)

model = YOLO(
    str(MODEL_PATH)
)

print(
    "YOLOv8m model loaded successfully."
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(
    image_path,
    confidence=0.1
):
    """
    Run YOLOv8m detection on an X-ray image.

    Parameters
    ----------
    image_path : str or Path
        Path to the X-ray image.

    confidence : float
        Minimum detection confidence threshold.

    Returns
    -------
    dict
        JSON-compatible detection result.
    """

    image_path = Path(
        image_path
    )

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Run YOLO inference
    # --------------------------------------------------------

    results = model(
        str(image_path),
        conf=confidence,
        verbose=False
    )

    detections = []

    # --------------------------------------------------------
    # Process detections
    # --------------------------------------------------------

    for result in results:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            score = float(
                box.conf[0]
            )

            # Use model's class name when available
            if class_id in result.names:

                class_name = result.names[
                    class_id
                ]

            elif class_id < len(CLASS_NAMES):

                class_name = CLASS_NAMES[
                    class_id
                ]

            else:

                class_name = "Unknown"

            # Bounding box coordinates
            x1, y1, x2, y2 = [
                float(value)
                for value in box.xyxy[0]
            ]

            detections.append(
                {
                    "class_id": class_id,

                    "class": class_name,

                    "confidence": round(
                        score,
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
                        ),
                    },
                }
            )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "model": "YOLOv8m",

        "image": str(
            image_path
        ),

        "detections": detections,

        "num_detections": len(
            detections
        ),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 60
    )

    print(
        "YOLOv8m REAL INFERENCE TEST"
    )

    print(
        "=" * 60
    )

    # Test image
    test_image = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "vinbigdata"
        / "train"
        / "000434271f63a053c4128a0ba6352c7f.png"
    )

    print(
        f"\nTest image:\n{test_image}"
    )

    output = predict_image(
        test_image,
        confidence=0.1
    )

    print(
        "\nPrediction result:"
    )

    print(
        output
    )