import json

import numpy as np
import pytest
from PIL import Image, ImageEnhance, ImageFilter

from src.monitoring import PredictionRecord, compute_image_stats

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