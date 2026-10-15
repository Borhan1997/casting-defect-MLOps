"""Monitoring helpers: the per-request prediction record.

Kept free of FastAPI and training imports, like inference.py.
"""

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
from PIL import Image


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_request_id() -> str:
    return uuid.uuid4().hex


@dataclass
class PredictionRecord:
    """One logged prediction. Field names are shared by the logger,
    the Prometheus metrics and the drift script."""

    label: str  # "def_front" or "ok_front"
    probability: float  # P(defect)
    width: int  # original image width
    height: int  # original image height
    brightness: float
    contrast: float
    sharpness: float
    timestamp: str = field(default_factory=_now_iso)
    request_id: str = field(default_factory=_new_request_id)

    @classmethod
    def create(
        cls,
        label: str,
        probability: float,
        width: int,
        height: int,
        brightness: float,
        contrast: float,
        sharpness: float,
    ) -> "PredictionRecord":
        """Build a record; timestamp and request_id are filled automatically."""
        return cls(
            label=label,
            probability=float(probability),
            width=int(width),
            height=int(height),
            brightness=float(brightness),
            contrast=float(contrast),
            sharpness=float(sharpness),
        )

    def to_dict(self) -> dict:
        """Plain dict that json.dumps accepts."""
        return asdict(self)


def compute_image_stats(image: Image.Image) -> dict:
    """Cheap drift signals computed on the original image (before any
    resize or normalization). Used by both the API and the reference script.

    Returns plain Python floats:
    - brightness: mean grayscale value (0-255)
    - contrast: standard deviation of grayscale values
    - sharpness: variance of the Laplacian (interior pixels only)
    """
    gray = np.asarray(image.convert("L"), dtype=np.float64)

    brightness = float(gray.mean())
    contrast = float(gray.std())

    if gray.shape[0] < 3 or gray.shape[1] < 3:
        sharpness = 0.0
    else:
        laplacian = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4 * gray[1:-1, 1:-1]
        )
        sharpness = float(laplacian.var())

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
    }