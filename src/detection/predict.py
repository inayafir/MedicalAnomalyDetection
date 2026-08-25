from pathlib import Path
from ultralytics import YOLO


# Path to the currently trained 4-class YOLO model
MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "detect"
    / "outputs"
    / "detection"
    / "yolov8n_4class"
    / "weights"
    / "best.pt"
)

# Load model once when this module is imported
model = YOLO(str(MODEL_PATH))


def predict_image(image_path, confidence=0.1):
    """
    Run YOLO detection on an X-ray image.

    Parameters
    ----------
    image_path : str or Path
        Path to the X-ray image.

    confidence : float
        Minimum confidence threshold.

    Returns
    -------
    dict
        JSON-compatible detection result.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model(
        str(image_path),
        conf=confidence,
        verbose=False
    )

    detections = []

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            score = float(box.conf[0])

            x1, y1, x2, y2 = [
                float(value) for value in box.xyxy[0]
            ]

            detections.append(
                {
                    "class_id": class_id,
                    "class": class_name,
                    "confidence": round(score, 4),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                }
            )

    return {
        "model": "YOLOv8n",
        "detections": detections,
        "num_detections": len(detections),
    }


if __name__ == "__main__":

    # Test image
    test_image = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "detection"
        / "yolo"
        / "images"
        / "val"
        / "0007d316f756b3fa0baea2ff514ce945.png"
    )

    output = predict_image(test_image)

    print("\nYOLO REAL INFERENCE")
    print("=" * 50)
    print(output)