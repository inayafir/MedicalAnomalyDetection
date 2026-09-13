import io
import os
import tempfile

from pathlib import Path

from app.config import settings


class TestServeFile:
    def test_serve_uploaded_image(self, client, sample_image_upload):
        filename, content, content_type = sample_image_upload
        resp = client.post(
            "/images/upload",
            files={"file": (filename, io.BytesIO(content), content_type)},
        )
        file_path = resp.json()["file_path"]

        resp = client.get(f"/files/{file_path}")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert len(resp.content) > 0

    def test_file_not_found(self, client):
        resp = client.get("/files/nonexistent/file.png")
        assert resp.status_code == 404

    def test_path_traversal_rejected(self, client):
        resp = client.get("/files/../../etc/passwd")
        assert resp.status_code in (400, 403, 404)

    def test_path_traversal_encoded(self, client):
        resp = client.get("/files/..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code in (400, 403, 404)

    def test_serve_heatmap(self, client, uploaded_image_with_prediction):
        image_id, prediction_id = uploaded_image_with_prediction
        # Get the prediction to find heatmap_path
        pred_resp = client.get(f"/predictions/{prediction_id}")
        heatmap_path = pred_resp.json()["heatmap_path"]

        if heatmap_path:
            resp = client.get(f"/files/{heatmap_path}")
            assert resp.status_code == 200
            assert "image" in resp.headers["content-type"]
