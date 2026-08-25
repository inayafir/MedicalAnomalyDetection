import io

import pytest

from app.models import FindingClass


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
        assert data["predicted_class"] in [c.value for c in FindingClass]
        assert 0 <= data["confidence"] <= 1
        assert isinstance(data["bboxes"], list)
        assert "heatmap_path" in data
        assert "created_at" in data

    def test_prediction_bbox_structure(self, client):
        image_id = self._upload_image(client)
        resp = client.post(f"/predictions/{image_id}")
        data = resp.json()
        if data["predicted_class"] == FindingClass.NORMAL.value:
            assert data["bboxes"] == []
        else:
            for bbox in data["bboxes"]:
                assert "class" in bbox
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

    def test_ml_interface_direct(self):
        from app.ml_interface import predict, is_model_loaded
        # Create a temp image
        img_bytes = io.BytesIO()
        from PIL import Image
        Image.new("RGB", (200, 200), (100, 100, 100)).save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Save to storage
        from app.storage import save_image
        from fastapi import UploadFile
        import io as _io

        class FakeUpload:
            def __init__(self, data, filename, content_type):
                self.file = _io.BytesIO(data)
                self.filename = filename
                self.content_type = content_type

        upload = FakeUpload(img_bytes.getvalue(), "test.png", "image/png")
        rel_path = save_image(upload)

        result = predict(rel_path, 200, 200)
        assert "class" in result
        assert "confidence" in result
        assert "bboxes" in result
        assert "heatmap_path" in result
        assert result["class"] in [c.value for c in FindingClass]
        assert 0 <= result["confidence"] <= 1
