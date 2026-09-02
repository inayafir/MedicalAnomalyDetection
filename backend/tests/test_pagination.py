import io


def _upload_image(client, name="x.png"):
    img_bytes = io.BytesIO()
    from PIL import Image
    Image.new("RGB", (64, 64), (128, 128, 128)).save(img_bytes, format="PNG")
    img_bytes.seek(0)
    resp = client.post(
        "/images/upload",
        files={"file": (name, img_bytes, "image/png")},
    )
    return resp.json()["id"]


class TestImagePagination:
    def test_empty_list(self, client):
        resp = client.get("/images")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["limit"] == 20
        assert data["offset"] == 0

    def test_default_pagination(self, client):
        for i in range(5):
            _upload_image(client, name=f"img{i}.png")
        resp = client.get("/images")
        data = resp.json()
        assert data["total"] == 5
        assert len(data["items"]) == 5
        # ordered by uploaded_at desc
        ids = [item["id"] for item in data["items"]]
        assert ids == sorted(ids, reverse=True)

    def test_limit_offset(self, client):
        ids = [_upload_image(client, name=f"img{i}.png") for i in range(5)]
        resp = client.get("/images?limit=2&offset=0")
        data = resp.json()
        assert len(data["items"]) == 2
        assert data["total"] == 5
        assert data["limit"] == 2
        assert data["offset"] == 0

        resp2 = client.get("/images?limit=2&offset=2")
        data2 = resp2.json()
        assert len(data2["items"]) == 2

        resp3 = client.get("/images?limit=2&offset=4")
        data3 = resp3.json()
        assert len(data3["items"]) == 1

    def test_limit_bounds(self, client):
        resp = client.get("/images?limit=0")
        assert resp.status_code == 422
        resp = client.get("/images?limit=101")
        assert resp.status_code == 422

    def test_offset_bounds(self, client):
        resp = client.get("/images?offset=-1")
        assert resp.status_code == 422

    def test_patient_id_filter(self, client):
        # Create a patient
        patient_resp = client.post("/patients", json={"display_name": "Test Patient"})
        patient_id = patient_resp.json()["id"]

        # Upload images: one with patient, one without
        img1_id = _upload_image(client, name="with_patient.png")
        client.put(f"/images/{img1_id}")  # won't work, but we just need the image

        # Upload with patient_id
        img_bytes = io.BytesIO()
        from PIL import Image
        Image.new("RGB", (64, 64), (128, 128, 128)).save(img_bytes, format="PNG")
        img_bytes.seek(0)
        resp = client.post(
            f"/images/upload?patient_id={patient_id}",
            files={"file": ("patient_img.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 201

        _upload_image(client, name="no_patient.png")

        # Filter by patient_id
        resp = client.get(f"/images?patient_id={patient_id}")
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["patient_id"] == patient_id


class TestPredictionPagination:
    def _create_prediction(self, client):
        img_bytes = io.BytesIO()
        from PIL import Image
        Image.new("RGB", (64, 64), (128, 128, 128)).save(img_bytes, format="PNG")
        img_bytes.seek(0)
        upload = client.post(
            "/images/upload",
            files={"file": ("x.png", img_bytes, "image/png")},
        )
        image_id = upload.json()["id"]
        pred = client.post(f"/predictions/{image_id}")
        return pred.json()["id"]

    def test_empty_list(self, client):
        resp = client.get("/predictions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_default_pagination(self, client):
        for _ in range(3):
            self._create_prediction(client)
        resp = client.get("/predictions")
        data = resp.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3
        # ordered by created_at desc
        ids = [item["id"] for item in data["items"]]
        assert ids == sorted(ids, reverse=True)

    def test_limit_offset(self, client):
        for _ in range(5):
            self._create_prediction(client)
        resp = client.get("/predictions?limit=2&offset=0")
        data = resp.json()
        assert len(data["items"]) == 2
        assert data["total"] == 5

    def test_image_id_filter(self, client):
        img_bytes = io.BytesIO()
        from PIL import Image
        Image.new("RGB", (64, 64), (128, 128, 128)).save(img_bytes, format="PNG")
        img_bytes.seek(0)
        upload = client.post(
            "/images/upload",
            files={"file": ("x.png", img_bytes, "image/png")},
        )
        image_id = upload.json()["id"]
        client.post(f"/predictions/{image_id}")
        client.post(f"/predictions/{image_id}")

        resp = client.get(f"/predictions?image_id={image_id}")
        data = resp.json()
        assert data["total"] == 2

    def test_predicted_class_filter(self, client):
        for _ in range(3):
            self._create_prediction(client)
        # Filter with a valid class
        resp = client.get("/predictions?predicted_class=Normal")
        assert resp.status_code == 200

        # Filter with invalid class
        resp = client.get("/predictions?predicted_class=InvalidClass")
        assert resp.status_code == 422

    def test_list_items_shape(self, client):
        self._create_prediction(client)
        resp = client.get("/predictions")
        item = resp.json()["items"][0]
        # List item should NOT have bboxes or heatmap_path (compact)
        assert "id" in item
        assert "image_id" in item
        assert "predicted_class" in item
        assert "confidence" in item
        assert "created_at" in item
        assert "bboxes" not in item
        assert "heatmap_path" not in item
