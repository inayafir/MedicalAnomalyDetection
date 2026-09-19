import io


class TestCreateReport:
    def _create_prediction(self, client):
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
        return pred.json()["id"]

    def test_report_created(self, client):
        prediction_id = self._create_prediction(client)
        resp = client.post(f"/reports/{prediction_id}")
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["prediction_id"] == prediction_id
        assert data["pdf_path"] is not None
        assert data["pdf_path"].endswith(".pdf")
        assert "generated_at" in data

    def test_report_pdf_is_downloadable(self, client):
        prediction_id = self._create_prediction(client)
        resp = client.post(f"/reports/{prediction_id}")
        pdf_path = resp.json()["pdf_path"]

        download = client.get(f"/files/{pdf_path}")
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/pdf"
        assert download.content.startswith(b"%PDF")

    def test_report_404_nonexistent_prediction(self, client):
        resp = client.post("/reports/9999")
        assert resp.status_code == 404
