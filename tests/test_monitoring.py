import io
import json
import logging

import numpy as np
import pytest
from PIL import Image, ImageEnhance, ImageFilter

from src.monitoring import (
    LOG_PATH_ENV,
    PredictionRecord,
    compute_image_stats,
    log_prediction,
)

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


def test_prediction_record_serializes_to_json():
    record = PredictionRecord.create(
        label="def_front",
        probability=0.91,
        width=300,
        height=300,
        brightness=112.5,
        contrast=40.2,
        sharpness=85.0,
    )

    payload = json.loads(json.dumps(record.to_dict()))

    assert set(payload) == EXPECTED_KEYS
    assert payload["label"] == "def_front"
    assert payload["request_id"]
    assert payload["timestamp"]


def _noise_image(size: int = 64, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    return Image.fromarray(pixels, mode="L")


def test_constant_gray_image_stats():
    stats = compute_image_stats(Image.new("L", (32, 32), 128))

    assert stats["brightness"] == pytest.approx(128.0)
    assert stats["contrast"] == pytest.approx(0.0)
    assert stats["sharpness"] == pytest.approx(0.0)


def test_darkened_image_has_lower_brightness():
    original = _noise_image()
    darkened = ImageEnhance.Brightness(original).enhance(0.5)

    assert (
        compute_image_stats(darkened)["brightness"]
        < compute_image_stats(original)["brightness"]
    )


def test_blurred_image_has_lower_sharpness():
    original = _noise_image()
    blurred = original.filter(ImageFilter.GaussianBlur(radius=2))

    assert (
        compute_image_stats(blurred)["sharpness"]
        < compute_image_stats(original)["sharpness"]
    )


def test_stats_are_plain_floats():
    stats = compute_image_stats(_noise_image().convert("RGB"))

    assert set(stats) == {"brightness", "contrast", "sharpness"}
    assert all(type(value) is float for value in stats.values())


FAKE_RESULT = {"label": "def_front", "probability": 0.93}


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def test_logging_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(LOG_PATH_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    log_prediction(_png_bytes(_noise_image()), FAKE_RESULT)

    assert list(tmp_path.iterdir()) == []


def test_log_prediction_appends_json_lines(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "predictions.jsonl"  # folder does not exist yet
    monkeypatch.setenv(LOG_PATH_ENV, str(log_path))
    png = _png_bytes(Image.new("RGB", (40, 30), (10, 20, 30)))

    log_prediction(png, FAKE_RESULT)
    log_prediction(png, FAKE_RESULT)

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 2
    for record in records:
        assert set(record) == EXPECTED_KEYS
        assert record["label"] == "def_front"
        assert record["width"] == 40
        assert record["height"] == 30
    assert records[0]["request_id"] != records[1]["request_id"]


def test_log_prediction_never_raises_on_unwritable_path(
    tmp_path, monkeypatch, caplog
):
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")  # a file where a directory is needed
    monkeypatch.setenv(LOG_PATH_ENV, str(blocker / "log.jsonl"))

    with caplog.at_level(logging.WARNING):
        log_prediction(_png_bytes(_noise_image()), FAKE_RESULT)

    assert _warnings(caplog)


def test_log_prediction_never_raises_on_bad_image(tmp_path, monkeypatch, caplog):
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setenv(LOG_PATH_ENV, str(log_path))

    with caplog.at_level(logging.WARNING):
        log_prediction(b"not an image", FAKE_RESULT)

    assert not log_path.exists()
    assert _warnings(caplog)