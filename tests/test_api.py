"""
pytest tests for the emotion recognition FastAPI endpoint.
Run: python -m pytest tests/ -v
All 9 tests must pass before opening any PR that touches src/app.py.
"""

import io
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch

# Patch model loading before import
with patch("src.app.load_model"):
    from src.app import app

client = TestClient(app)

LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def make_fake_image(h=100, w=100, channels=3):
    """Create a minimal in-memory JPEG for testing."""
    import cv2
    img = np.random.randint(0, 255, (h, w, channels), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    return io.BytesIO(encoded.tobytes())


# --- Health endpoint ---

def test_health_returns_200():
    """GET /health must return HTTP 200."""
    r = client.get("/health")
    assert r.status_code == 200


def test_health_has_status_field():
    """GET /health response must include a status field."""
    r = client.get("/health")
    assert "status" in r.json()


def test_health_status_is_ok():
    """GET /health status value must be 'ok'."""
    r = client.get("/health")
    assert r.json()["status"] == "ok"


# --- Classes endpoint ---

def test_classes_returns_seven_labels():
    """GET /classes must return exactly 7 emotion labels."""
    r = client.get("/classes")
    assert r.status_code == 200
    data = r.json()
    assert "classes" in data
    assert len(data["classes"]) == 7


def test_classes_contains_happy():
    """GET /classes must include 'Happy' as a class."""
    r = client.get("/classes")
    assert "Happy" in r.json()["classes"]


# --- Predict endpoint ---

def test_predict_returns_200_with_valid_image():
    """POST /predict with a valid JPEG must return HTTP 200."""
    img = make_fake_image()
    r = client.post("/predict", files={"file": ("test.jpg", img, "image/jpeg")})
    assert r.status_code == 200


def test_predict_response_has_required_fields():
    """POST /predict response must include faces_detected and detections fields."""
    img = make_fake_image()
    r = client.post("/predict", files={"file": ("test.jpg", img, "image/jpeg")})
    data = r.json()
    assert "faces_detected" in data
    assert "detections" in data


def test_predict_faces_detected_is_integer():
    """POST /predict faces_detected field must be an integer."""
    img = make_fake_image()
    r = client.post("/predict", files={"file": ("test.jpg", img, "image/jpeg")})
    assert isinstance(r.json()["faces_detected"], int)


def test_predict_returns_422_for_empty_file():
    """POST /predict with an empty file must return HTTP 422."""
    r = client.post(
        "/predict",
        files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")}
    )
    assert r.status_code == 422
