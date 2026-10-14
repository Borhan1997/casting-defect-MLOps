"""API tests, run with: pytest tests/test_api.py -v

Needs models/baseline.pt and two real casting images in tests/fixtures/:
    def_front.jpg  (a defective part)
    ok_front.jpg   (a good part)
"""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

pytestmark = pytest.mark.skipif(
    not Path("models/baseline.pt").exists(),
    reason="models/baseline.pt not available (pull it with DVC)",
)

from src.api import app # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def client():
    # The context-manager form runs the lifespan handler, so the model is loaded at startup.
    with TestClient(app) as c:
        yield c


def _post_image(client: TestClient, content: bytes, filename: str = "test.jpg",
                content_type: str = "image/jpeg"):
    return client.post("/predict", files={"file": (filename, content, content_type)})


def _synthetic_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), color=(128, 128, 128)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_structure(client):
    response = _post_image(client, _synthetic_jpeg())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"label", "probability"}
    assert body["label"] in {"def_front", "ok_front"}
    assert 0.0 <= body["probability"] <= 1.0


@pytest.mark.parametrize(
    "filename, expected_label",
    [("def_front.jpg", "def_front"), ("ok_front.jpg", "ok_front")],
)
def test_predict_real_images(client, filename, expected_label):
    image_bytes = (FIXTURES / filename).read_bytes()
    response = _post_image(client, image_bytes, filename=filename)
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == expected_label
    # Label must be consistent with the probability (threshold 0.5)
    assert (body["probability"] >= 0.5) == (expected_label == "def_front")


def test_predict_rejects_non_image(client):
    response = _post_image(client, b"this is not an image", filename="notes.txt",
                           content_type="text/plain")
    assert response.status_code == 400
    assert response.json()["detail"] == "File is not a readable image."


def test_predict_rejects_empty_file(client):
    response = _post_image(client, b"", filename="empty.jpg")
    assert response.status_code == 400
    assert response.json()["detail"] == "Empty file."


def test_predict_without_file_is_422(client):
    response = client.post("/predict")
    assert response.status_code == 422