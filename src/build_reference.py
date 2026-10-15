"""Build the reference profile used for input-drift monitoring.

Offline script: it is never copied into the serving image.

Usage (from the project root):
    python -m src.build_reference \
        --images-dir data/processed/train \
        --val-manifest data/val_manifest.txt

The reference is what the deployed model was trained on, so the validation
images listed in the manifest are excluded.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from src.monitoring import compute_image_stats

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FEATURES = ("brightness", "contrast", "sharpness")
PERCENTILE_POINTS = list(range(0, 101, 10))  # 0, 10, ..., 100 -> 11 values
DEFAULT_OUTPUT = "monitoring/reference_profile.json"


def load_manifest(path: str | Path) -> set[str]:
    """Read manifest keys, one per line, e.g. 'def_front/xyz.jpg'."""
    lines = Path(path).read_text().splitlines()
    return {line.strip() for line in lines if line.strip()}


def build_reference(images_dir: Path, exclude: set[str] | None = None) -> dict:
    """Compute the reference distribution of image statistics.

    Images whose POSIX path relative to `images_dir` is in `exclude` are skipped.
    """
    images_dir = Path(images_dir)
    exclude = exclude or set()

    values: dict[str, list[float]] = {name: [] for name in FEATURES}
    n_excluded = 0

    for path in sorted(images_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.relative_to(images_dir).as_posix() in exclude:
            n_excluded += 1
            continue
        with Image.open(path) as image:
            stats = compute_image_stats(image)
        for name in FEATURES:
            values[name].append(stats[name])

    n_images = len(values[FEATURES[0]])
    if n_images == 0:
        raise ValueError(f"no usable images found in {images_dir}")

    profile: dict = {
        "n_images": n_images,
        "n_excluded": n_excluded,
        "source": str(images_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    for name in FEATURES:
        array = np.asarray(values[name], dtype=np.float64)
        profile[name] = {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "percentiles": np.percentile(array, PERCENTILE_POINTS).tolist(),
        }
    return profile


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the drift reference profile")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument(
        "--val-manifest",
        default=None,
        help="file listing validation images (relative paths) to exclude",
    )
    args = parser.parse_args(argv)

    exclude = load_manifest(args.val_manifest) if args.val_manifest else None
    profile = build_reference(args.images_dir, exclude)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n")

    print(
        f"Reference written to {args.output}: "
        f"{profile['n_images']} images used, {profile['n_excluded']} excluded"
    )


if __name__ == "__main__":
    main()