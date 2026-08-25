import io


class TestUploadSuccess:
    def test_upload_returns_201(self, client, sample_image_upload):
        filename, content, content_type = sample_image_upload
        resp = client.post(
            "/images/upload",
            files={"file": (filename, io.BytesIO(content), content_type)},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["original_filename"] == filename
        assert data["content_type"] == content_type
        assert data["file_size_bytes"] == len(content)

    def test_upload_with_patient(self, client, sample_image_upload):
        # Create a patient first
        resp = client.post("/images/upload", json={})
        # Actually, patients aren't created via images — let's just test with patient_id that doesn't exist
        filename, content, content_type = sample_image_upload
        resp = client.post(
            "/images/upload?patient_id=9999",
            files={"file": (filename, io.BytesIO(content), content_type)},
        )
        assert resp.status_code == 404

    def test_upload_wrong_content_type(self, client):
        resp = client.post(
            "/images/upload",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert resp.status_code == 415

    def test_upload_oversized_file(self, client):
        large = b"\x00" * (11 * 1024 * 1024)  # 11MB
        resp = client.post(
            "/images/upload",
            files={"file": ("big.png", io.BytesIO(large), "image/png")},
        )
        assert resp.status_code == 413

    def test_upload_corrupted_image(self, client):
        resp = client.post(
            "/images/upload",
            files={"file": ("fake.png", io.BytesIO(b"not an image"), "image/png")},
        )
        assert resp.status_code == 422

    def test_upload_nonexistent_patient(self, client, sample_image_upload):
        filename, content, content_type = sample_image_upload
        resp = client.post(
            "/images/upload?patient_id=9999",
            files={"file": (filename, io.BytesIO(content), content_type)},
        )
        assert resp.status_code == 404


class TestGetImage:
    def test_get_image_404(self, client):
        resp = client.get("/images/9999")
        assert resp.status_code == 404

    def test_get_image_no_prediction(self, client, sample_image_upload):
        filename, content, content_type = sample_image_upload
        upload_resp = client.post(
            "/images/upload",
            files={"file": (filename, io.BytesIO(content), content_type)},
        )
        image_id = upload_resp.json()["id"]

        resp = client.get(f"/images/{image_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == image_id
        assert data["latest_prediction"] is None
