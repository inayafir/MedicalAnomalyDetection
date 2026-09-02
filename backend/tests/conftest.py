import io
import random

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base, get_db
from app.main import app
from app.models import CLASSIFIER_CLASSES, DETECTOR_CLASSES

TEST_DB_URL = "sqlite://"  # in-memory


def _mock_predict(image_path: str, original_width: int, original_height: int) -> dict:
    """Mock predict that returns valid contract-shaped output for fast tests."""
    chosen_class = random.choice(CLASSIFIER_CLASSES)

    bboxes = []
    if chosen_class != "Normal":
        num_bboxes = random.randint(1, 3)
        for _ in range(num_bboxes):
            detector_class = random.choice(DETECTOR_CLASSES)
            bx1 = random.randint(0, original_width // 2)
            by1 = random.randint(0, original_height // 2)
            bx2 = random.randint(bx1 + 10, min(bx1 + 200, original_width))
            by2 = random.randint(by1 + 10, min(by1 + 200, original_height))
            bboxes.append({
                "class": detector_class,
                "x1": bx1,
                "y1": by1,
                "x2": bx2,
                "y2": by2,
                "confidence": round(random.uniform(0.60, 0.99), 2),
            })

    return {
        "class": chosen_class,
        "confidence": round(random.uniform(0.60, 0.99), 2),
        "bboxes": bboxes,
        "heatmap_path": None,
    }


@pytest.fixture(scope="session")
def _test_engine():
    engine = create_engine(
        TEST_DB_URL, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def db_session(_test_engine):
    connection = _test_engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture()
def client(db_session, monkeypatch):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    # Mock predict and is_model_loaded for fast tests (no real models needed)
    monkeypatch.setattr("app.routers.predictions.predict", _mock_predict)
    monkeypatch.setattr("app.routers.predictions.is_model_loaded", lambda: True)

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def sample_image_bytes():
    img = Image.new("RGB", (128, 128), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def sample_image_upload(sample_image_bytes):
    return ("test.png", sample_image_bytes, "image/png")


@pytest.fixture()
def uploaded_image_id(client, sample_image_upload):
    """Upload an image and return its ID for use in tests."""
    filename, content, content_type = sample_image_upload
    resp = client.post(
        "/images/upload",
        files={"file": (filename, io.BytesIO(content), content_type)},
    )
    return resp.json()["id"]


@pytest.fixture()
def uploaded_image_with_prediction(client, sample_image_upload):
    """Upload an image and create a prediction, return (image_id, prediction_id)."""
    filename, content, content_type = sample_image_upload
    resp = client.post(
        "/images/upload",
        files={"file": (filename, io.BytesIO(content), content_type)},
    )
    image_id = resp.json()["id"]
    pred_resp = client.post(f"/predictions/{image_id}")
    prediction_id = pred_resp.json()["id"]
    return image_id, prediction_id
