"""API-level tests for the Prometheus metrics, run with: pytest tests/test_api_metrics.py -v

Needs models/baseline.pt (skipped otherwise, like tests/test_api.py).
TestClient runs the app in-process, so the metrics registry can be read directly.
The registry is global to the process, so the tests compare deltas.
"""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from prometheus_client import REGISTRY

if not Path("models/baseline.pt").exists():
    pytest.skip(
        "models/baseline.pt not available (pull it with DVC)",
        allow_module_level=True,
    )

from src.api import app  # noqa: E402

LABELS = ("def_front", "ok_front")


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


def _count(label: str) -> float:
    value = REGISTRY.get_sample_value("cv_predictions_total", {"label": label})
    return value or 0.0


def test_metrics_endpoint_is_exposed(client):
    assert _post_image(client, _synthetic_jpeg()).status_code == 200

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "cv_predictions_total{label=" in response.text
    assert "cv_image_brightness_bucket" in response.text


def test_predict_increments_counter_for_the_returned_label(client):
    before = {label: _count(label) for label in LABELS}

    response = _post_image(client, _synthetic_jpeg())

    assert response.status_code == 200
    returned = response.json()["label"]
    other = next(label for label in LABELS if label != returned)
    assert _count(returned) == before[returned] + 1
    assert _count(other) == before[other]


def test_rejected_requests_are_not_counted(client):
    before = {label: _count(label) for label in LABELS}

    not_an_image = _post_image(client, b"not an image", filename="notes.txt",
                               content_type="text/plain")
    empty_file = _post_image(client, b"", filename="empty.jpg")

    assert not_an_image.status_code == 400
    assert empty_file.status_code == 400
    assert {label: _count(label) for label in LABELS} == before