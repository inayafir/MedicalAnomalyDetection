import io

import pytest

from app.models import CLASSIFIER_CLASSES, DETECTOR_CLASSES, FindingClass


class TestCreatePrediction:
    def _upload_image(self, client):
        img_bytes = io.BytesIO()
        from PIL import Image
        Image.new("RGB", (200, 200), (100, 100, 100)).save(img_bytes, format="PNG")
        img_bytes.seek(0)
        resp = client.post(
            "/images/upload",
            files={"file": ("x.png", img_bytes, "image/png")},
        )
        return resp.json()["id"]

    def test_prediction_created(self, client):
        image_id = self._upload_image(client)
        resp = client.post(f"/predictions/{image_id}")
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["image_id"] == image_id
        # Top-level class must be one of the 5 classifier classes
        assert data["predicted_class"] in CLASSIFIER_CLASSES
        assert data["predicted_class"] in [c.value for c in FindingClass]
        assert 0 <= data["confidence"] <= 1
        assert isinstance(data["bboxes"], list)
        assert "heatmap_path" in data
        assert "created_at" in data

    def test_prediction_bbox_structure(self, client):
        image_id = self._upload_image(client)
        resp = client.post(f"/predictions/{image_id}")
        data = resp.json()
        if data["predicted_class"] == "Normal":
            assert data["bboxes"] == []
        else:
            for bbox in data["bboxes"]:
                assert "class_" in bbox or "class" in bbox
                class_key = "class_" if "class_" in bbox else "class"
                # Bbox class must be one of the 14 detector classes
                assert bbox[class_key] in DETECTOR_CLASSES
                assert "x1" in bbox and "y1" in bbox
                assert "x2" in bbox and "y2" in bbox
                assert "confidence" in bbox
                assert bbox["x2"] > bbox["x1"]
                assert bbox["y2"] > bbox["y1"]

    def test_prediction_creates_two_rows(self, client):
        image_id = self._upload_image(client)
        r1 = client.post(f"/predictions/{image_id}")
        r2 = client.post(f"/predictions/{image_id}")
        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["id"] != r2.json()["id"]

    def test_prediction_404_nonexistent_image(self, client):
        resp = client.post("/predictions/9999")
        assert resp.status_code == 404


class TestGetPrediction:
    def test_get_prediction(self, client):
        from PIL import Image
        img_bytes = io.BytesIO()
        Image.new("RGB", (200, 200), (100, 100, 100)).save(img_bytes, format="PNG")
        img_bytes.seek(0)
        upload = client.post(
            "/images/upload",
            files={"file": ("x.png", img_bytes, "image/png")},
        )
        image_id = upload.json()["id"]
        pred = client.post(f"/predictions/{image_id}")
        pred_id = pred.json()["id"]

        resp = client.get(f"/predictions/{pred_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == pred_id

    def test_get_prediction_404(self, client):
        resp = client.get("/predictions/9999")
        assert resp.status_code == 404


@pytest.mark.integration
class TestRealModelPrediction:
    """Tests that exercise the real ML models. Slow — skip with -m 'not integration'."""

    def test_checkpoint_class_counts(self):
        """Verify checkpoint class counts match expectations at import time."""
        from app.models import CLASSIFIER_CLASSES, DETECTOR_CLASSES
        assert len(CLASSIFIER_CLASSES) == 5, f"Expected 5 classifier classes, got {len(CLASSIFIER_CLASSES)}"
        assert len(DETECTOR_CLASSES) == 14, f"Expected 14 detector classes, got {len(DETECTOR_CLASSES)}"

    def test_ml_interface_direct(self):
        from app.ml_interface import predict, is_model_loaded, load_models, get_classifier_classes, get_detector_classes

        # Ensure models are loaded (lifespan doesn't run in standalone tests)
        if not is_model_loaded():
            load_models()
        assert is_model_loaded(), "Models should be loaded for integration test"

        # Create a temp image
        img_bytes = io.BytesIO()
        from PIL import Image
        Image.new("RGB", (200, 200), (100, 100, 100)).save(img_bytes, format="PNG")
        img_bytes.seek(0)

        from app.storage import save_image

        class FakeUpload:
            def __init__(self, data, filename, content_type):
                self.file = io.BytesIO(data)
                self.filename = filename
                self.content_type = content_type

        upload = FakeUpload(img_bytes.getvalue(), "test.png", "image/png")
        rel_path = save_image(upload)

        result = predict(rel_path, 200, 200)

        # Top-level class from classifier (5 classes)
        classifier_classes = get_classifier_classes()
        assert result["class"] in classifier_classes, (
            f"Top-level class {result['class']!r} not in classifier classes: {classifier_classes}"
        )
        assert 0 <= result["confidence"] <= 1

        # Bbox classes from detector (14 classes)
        detector_classes = get_detector_classes()
        for bbox in result["bboxes"]:
            assert bbox["class"] in detector_classes, (
                f"Bbox class {bbox['class']!r} not in detector classes: {detector_classes}"
            )
            assert 0 <= bbox["confidence"] <= 1
            assert bbox["x1"] < bbox["x2"]
            assert bbox["y1"] < bbox["y2"]
            # Coordinates should be within original image bounds
            assert 0 <= bbox["x1"] < 200
            assert 0 <= bbox["y1"] < 200
            assert bbox["x2"] <= 200
            assert bbox["y2"] <= 200

        assert "heatmap_path" in result
