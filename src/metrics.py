"""Prometheus metrics for the prediction service.

Imports prometheus_client, Pillow and src.monitoring only (no FastAPI, no torch).

The bucket edges of the three image-statistic histograms are the deciles of the
reference profile (monitoring/reference_profile.json). On clean traffic each
bucket then receives about 10% of the observations, so a darkened or blurred
stream shows up as a pile-up in the lowest buckets without any statistics.
"""

import io
import json
import logging
import math
import os
from pathlib import Path

from PIL import Image
from prometheus_client import Counter, Histogram

from src.monitoring import compute_image_stats

logger = logging.getLogger(__name__)

PROFILE_PATH_ENV = "REFERENCE_PROFILE_PATH"
DEFAULT_PROFILE_PATH = "monitoring/reference_profile.json"
FEATURES = ("brightness", "contrast", "sharpness")

# Used when the reference profile is missing or unusable.
DEFAULT_BUCKETS = {
    "brightness": [25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0],
    "contrast": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 128.0],
    "sharpness": [25.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 600.0, 1000.0],
}

LATENCY_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]

# Dense near the ends: the model is very confident, so most of the mass sits
# below 0.1 or above 0.9.
PROBABILITY_BUCKETS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]


def load_buckets(path: str | Path) -> dict[str, list[float]]:
    """Bucket edges per feature, taken from the reference percentiles.

    Never raises: a missing file, malformed JSON or a missing or invalid
    feature falls back to DEFAULT_BUCKETS for that feature, so the API can
    always start.
    """
    buckets = {name: list(edges) for name, edges in DEFAULT_BUCKETS.items()}

    try:
        profile = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return buckets

    for name in FEATURES:
        try:
            values = {float(v) for v in profile[name]["percentiles"]}
        except (KeyError, TypeError, ValueError):
            continue
        edges = sorted(v for v in values if math.isfinite(v))
        if edges:
            buckets[name] = edges
    return buckets


_BUCKETS = load_buckets(os.environ.get(PROFILE_PATH_ENV, DEFAULT_PROFILE_PATH))

PREDICTIONS = Counter(
    "cv_predictions_total",
    "Predictions served, by predicted label.",
    ["label"],
)
LATENCY = Histogram(
    "cv_prediction_latency_seconds",
    "Time spent in predict().",
    buckets=LATENCY_BUCKETS,
)
PROBABILITY = Histogram(
    "cv_defect_probability",
    "Predicted probability of a defect.",
    buckets=PROBABILITY_BUCKETS,
)
BRIGHTNESS = Histogram(
    "cv_image_brightness",
    "Mean grayscale value of incoming images.",
    buckets=_BUCKETS["brightness"],
)
CONTRAST = Histogram(
    "cv_image_contrast",
    "Grayscale standard deviation of incoming images.",
    buckets=_BUCKETS["contrast"],
)
SHARPNESS = Histogram(
    "cv_image_sharpness",
    "Laplacian variance of incoming images.",
    buckets=_BUCKETS["sharpness"],
)


def observe_prediction(result: dict, stats: dict, latency_seconds: float) -> None:
    """Record one served prediction (single entry point for the API)."""
    PREDICTIONS.labels(label=result["label"]).inc()
    PROBABILITY.observe(result["probability"])
    BRIGHTNESS.observe(stats["brightness"])
    CONTRAST.observe(stats["contrast"])
    SHARPNESS.observe(stats["sharpness"])
    LATENCY.observe(latency_seconds)


def record_prediction(image_bytes: bytes, result: dict, latency_seconds: float) -> None:
    """Compute the image statistics and record one served prediction. Never raises.

    Like log_prediction, this is monitoring: it is allowed to fail, serving is not.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            stats = compute_image_stats(image)
        observe_prediction(result, stats, latency_seconds)
    except Exception:
        logger.warning("could not record prediction metrics", exc_info=True)