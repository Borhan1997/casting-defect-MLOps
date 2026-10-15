"""API-level tests for prediction logging, run with: pytest tests/test_api_logging.py -v

Needs models/baseline.pt (skipped otherwise, like tests/test_api.py).
Logging is opt-in through the PREDICTION_LOG_PATH env var, which the app reads
at call time, so each test can point it somewhere different.
"""
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

if not Path("models/baseline.pt").exists():
    pytest.skip(
        "models/baseline.pt not available (pull it with DVC)",
        allow_module_level=True,
    )

from src.api import app  # noqa: E402
from src.monitoring import LOG_PATH_ENV  # noqa: E402

EXPECTED_KEYS = {
    "timestamp",
    "request_id",
    "label",
    "probability",
    "width",
    "height",
    "brightness",
    "contrast",
    "sharpness",
}


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


def _read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_predict_writes_one_log_line(client, tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "predictions.jsonl"  # folder does not exist yet
    monkeypatch.setenv(LOG_PATH_ENV, str(log_path))

    response = _post_image(client, _synthetic_jpeg())

    assert response.status_code == 200
    body = response.json()
    records = _read_log(log_path)
    assert len(records) == 1
    record = records[0]
    assert set(record) == EXPECTED_KEYS
    assert record["width"] == 300
    assert record["height"] == 300
    assert record["label"] == body["label"]
    assert record["probability"] == pytest.approx(body["probability"])


def test_logging_off_creates_nothing(client, tmp_path, monkeypatch):
    monkeypatch.delenv(LOG_PATH_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    response = _post_image(client, _synthetic_jpeg())

    assert response.status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_logging_failure_does_not_break_the_response(client, tmp_path, monkeypatch):
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")  # a file where a directory is needed
    monkeypatch.setenv(LOG_PATH_ENV, str(blocker / "predictions.jsonl"))

    response = _post_image(client, _synthetic_jpeg())

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"def_front", "ok_front"}
    assert 0.0 <= body["probability"] <= 1.0


def test_rejected_requests_are_not_logged(client, tmp_path, monkeypatch):
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setenv(LOG_PATH_ENV, str(log_path))

    not_an_image = _post_image(client, b"not an image", filename="notes.txt",
                               content_type="text/plain")
    empty_file = _post_image(client, b"", filename="empty.jpg")

    assert not_an_image.status_code == 400
    assert empty_file.status_code == 400
    assert not log_path.exists()