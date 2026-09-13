import io
import os

from pathlib import Path

from app.config import settings


class TestDeleteImage:
    def _upload_image(self, client):
        img_bytes = io.BytesIO()
        from PIL import Image
        Image.new("RGB", (64, 64), (128, 128, 128)).save(img_bytes, format="PNG")
        img_bytes.seek(0)
        resp = client.post(
            "/images/upload",
            files={"file": ("del_test.png", img_bytes, "image/png")},
        )
        return resp.json()

    def test_delete_image_204(self, client):
        img_data = self._upload_image(client)
        image_id = img_data["id"]
        file_path = img_data["file_path"]

        # Verify file exists
        abs_path = Path(settings.STORAGE_ROOT) / file_path
        assert abs_path.exists()

        resp = client.delete(f"/images/{image_id}")
        assert resp.status_code == 204

        # Verify file is deleted
        assert not abs_path.exists()

        # Verify DB row is gone
        resp = client.get(f"/images/{image_id}")
        assert resp.status_code == 404

    def test_delete_cascades_to_predictions(self, client):
        img_data = self._upload_image(client)
        image_id = img_data["id"]

        # Create a prediction
        pred_resp = client.post(f"/predictions/{image_id}")
        assert pred_resp.status_code == 201
        prediction_id = pred_resp.json()["id"]

        # Delete image
        resp = client.delete(f"/images/{image_id}")
        assert resp.status_code == 204

        # Verify prediction is gone
        pred_resp = client.get(f"/predictions/{prediction_id}")
        assert pred_resp.status_code == 404

    def test_delete_cascades_heatmap_files(self, client):
        img_data = self._upload_image(client)
        image_id = img_data["id"]

        # Create a prediction (generates a heatmap)
        pred_resp = client.post(f"/predictions/{image_id}")
        heatmap_path = pred_resp.json()["heatmap_path"]

        if heatmap_path:
            heatmap_abs = Path(settings.STORAGE_ROOT) / heatmap_path
            assert heatmap_abs.exists()

            client.delete(f"/images/{image_id}")
            assert not heatmap_abs.exists()

    def test_delete_nonexistent_404(self, client):
        resp = client.delete("/images/9999")
        assert resp.status_code == 404

    def test_delete_multiple_images(self, client):
        ids = []
        for _ in range(3):
            img_data = self._upload_image(client)
            ids.append(img_data["id"])

        for image_id in ids:
            resp = client.delete(f"/images/{image_id}")
            assert resp.status_code == 204

        resp = client.get("/images")
        assert resp.json()["total"] == 0
