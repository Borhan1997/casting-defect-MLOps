"""Monitoring helpers: the per-request prediction record.

Kept free of FastAPI and training imports, like inference.py.
"""

import io
import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

LOG_PATH_ENV = "PREDICTION_LOG_PATH"

logger = logging.getLogger(__name__)
_log_lock = threading.Lock()


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


def get_log_path() -> Path | None:
    """Log file path from PREDICTION_LOG_PATH, or None when logging is off.

    Read at call time (not import time) so tests can set it with monkeypatch.
    """
    value = os.environ.get(LOG_PATH_ENV, "").strip()
    return Path(value) if value else None


def append_record(record: PredictionRecord, path: Path) -> None:
    """Append one record as a JSON line. The lock keeps lines from interleaving."""
    line = json.dumps(record.to_dict())
    path.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def log_prediction(image_bytes: bytes, result: dict) -> None:
    """Log one prediction if PREDICTION_LOG_PATH is set. Never raises.

    Monitoring is allowed to fail; serving is not. Any error is reported as a
    warning and the caller carries on.
    """
    path = get_log_path()
    if path is None:
        return

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            stats = compute_image_stats(image)
        record = PredictionRecord.create(
            label=result["label"],
            probability=result["probability"],
            width=width,
            height=height,
            **stats,
        )
        append_record(record, path)
    except Exception:
        logger.warning("could not log prediction", exc_info=True)