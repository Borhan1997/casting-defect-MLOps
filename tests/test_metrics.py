import io
import json
import logging

import pytest
from PIL import Image
from prometheus_client import REGISTRY

from src.metrics import (
    DEFAULT_BUCKETS,
    FEATURES,
    load_buckets,
    observe_prediction,
    record_prediction,
)


def _write_profile(path, **percentiles):
    profile = {
        name: {"mean": 0.0, "std": 1.0, "percentiles": values}
        for name, values in percentiles.items()
    }
    path.write_text(json.dumps(profile))
    return path


def _sample(name, labels=None):
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def test_default_buckets_are_strictly_increasing():
    for edges in DEFAULT_BUCKETS.values():
        assert edges == sorted(set(edges))


def test_load_buckets_returns_sorted_edges(tmp_path):
    path = _write_profile(
        tmp_path / "profile.json",
        brightness=[100.0, 20.0, 60.0, 80.0],
        contrast=[1.0, 2.0, 3.0],
        sharpness=[10.0, 20.0, 30.0],
    )

    buckets = load_buckets(path)

    assert buckets["brightness"] == [20.0, 60.0, 80.0, 100.0]
    assert buckets["brightness"][0] == 20.0
    assert buckets["brightness"][-1] == 100.0


def test_load_buckets_removes_repeated_values(tmp_path):
    path = _write_profile(
        tmp_path / "profile.json",
        brightness=[5.0, 5.0, 5.0, 10.0, 10.0, 20.0],
        contrast=[1.0, 2.0],
        sharpness=[1.0, 2.0],
    )

    edges = load_buckets(path)["brightness"]

    assert edges == [5.0, 10.0, 20.0]
    assert all(a < b for a, b in zip(edges, edges[1:]))


def test_missing_file_falls_back_to_defaults(tmp_path):
    assert load_buckets(tmp_path / "does_not_exist.json") == DEFAULT_BUCKETS


def test_malformed_json_falls_back_to_defaults(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("{not json")

    assert load_buckets(path) == DEFAULT_BUCKETS


def test_missing_feature_falls_back_for_that_feature_only(tmp_path):
    path = _write_profile(
        tmp_path / "profile.json",
        brightness=[10.0, 20.0],
        contrast=[1.0, 2.0],
    )

    buckets = load_buckets(path)

    assert buckets["brightness"] == [10.0, 20.0]
    assert buckets["contrast"] == [1.0, 2.0]
    assert buckets["sharpness"] == DEFAULT_BUCKETS["sharpness"]


@pytest.mark.parametrize("bad_value", [None, "abc", [], {}, 7])
def test_invalid_percentiles_fall_back_to_defaults(tmp_path, bad_value):
    path = _write_profile(
        tmp_path / "profile.json",
        brightness=bad_value,
        contrast=bad_value,
        sharpness=bad_value,
    )

    assert load_buckets(path) == DEFAULT_BUCKETS


def test_all_features_always_present(tmp_path):
    for buckets in (
        load_buckets(tmp_path / "nope.json"),
        load_buckets(_write_profile(tmp_path / "p.json", brightness=[1.0])),
    ):
        assert set(buckets) == set(FEATURES)


def test_observe_prediction_updates_every_metric():
    counted = [
        ("cv_predictions_total", {"label": "def_front"}),
        ("cv_defect_probability_count", None),
        ("cv_image_brightness_count", None),
        ("cv_image_contrast_count", None),
        ("cv_image_sharpness_count", None),
        ("cv_prediction_latency_seconds_count", None),
    ]
    before = {name: _sample(name, labels) for name, labels in counted}
    other_label_before = _sample("cv_predictions_total", {"label": "ok_front"})

    observe_prediction(
        {"label": "def_front", "probability": 0.97},
        {"brightness": 140.0, "contrast": 55.0, "sharpness": 200.0},
        0.05,
    )

    for name, labels in counted:
        assert _sample(name, labels) == before[name] + 1
    assert _sample("cv_predictions_total", {"label": "ok_front"}) == other_label_before


_COUNTED = [
    ("cv_predictions_total", {"label": "def_front"}),
    ("cv_defect_probability_count", None),
    ("cv_image_brightness_count", None),
    ("cv_image_contrast_count", None),
    ("cv_image_sharpness_count", None),
    ("cv_prediction_latency_seconds_count", None),
]
_RESULT = {"label": "def_front", "probability": 0.9}


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (120, 130, 140)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_record_prediction_updates_metrics():
    before = {name: _sample(name, labels) for name, labels in _COUNTED}

    record_prediction(_png_bytes(), _RESULT, 0.05)

    for name, labels in _COUNTED:
        assert _sample(name, labels) == before[name] + 1


def test_record_prediction_never_raises_on_bad_image(caplog):
    before = {name: _sample(name, labels) for name, labels in _COUNTED}

    with caplog.at_level(logging.WARNING):
        record_prediction(b"not an image", _RESULT, 0.05)

    assert any(r.levelno == logging.WARNING for r in caplog.records)
    for name, labels in _COUNTED:
        assert _sample(name, labels) == before[name]