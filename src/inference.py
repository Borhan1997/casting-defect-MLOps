"""Inference for the casting-defect classifier.

Loads the exported state_dict once and turns uploaded image bytes into a
label plus the probability of a defect.

Deliberately independent of the training code (no MLflow, no scikit-learn),
so the serving image stays small.
"""
import io
import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

# Where the exported model lives; override with the MODEL_PATH env var (e.g. in Docker).
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/baseline.pt"))

# Contract: label 1 = def_front (defect), label 0 = ok_front.
# `probability` is P(defect). A prediction is "def_front" when it reaches the threshold.
LABEL_DEFECT = "def_front"
LABEL_OK = "ok_front"
THRESHOLD = 0.5

# Same preprocessing as evaluation (resize 256, center-crop 224, ImageNet normalization).
# Accessing .transforms() does not download any weights.
_PREPROCESS = ResNet18_Weights.DEFAULT.transforms()


def _build_architecture() -> nn.Module:
    """ResNet18 with a 1-output head, matching src/model.py, but without pretrained weights."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


@lru_cache(maxsize=1)
def load_model() -> nn.Module:
    """Build the model and load the exported weights. Cached: runs once per process."""
    model = _build_architecture()
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(image_bytes: bytes) -> dict:
    """Classify one image given as raw bytes.

    Returns {"label": "def_front" | "ok_front", "probability": P(defect)}.
    Raises PIL.UnidentifiedImageError if the bytes are not a readable image.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    batch = _PREPROCESS(img).unsqueeze(0)  # (1, 3, 224, 224)

    model = load_model()
    with torch.inference_mode():
        logit = model(batch)  # (1, 1)
        prob_defect = torch.sigmoid(logit).item()

    label = LABEL_DEFECT if prob_defect >= THRESHOLD else LABEL_OK
    return {"label": label, "probability": prob_defect}