"""Replay traffic against the running API to show the drift monitoring react.

Offline script: it is never copied into the serving image.

Usage (the stack must be running: docker compose up -d):
    python -m src.simulate_drift --phases clean
    python -m src.simulate_drift --phases dark --per-phase 300 --rate 3
    python -m src.simulate_drift                  # all four phases, in order

Phases:
    clean    validation images, sent as they are (the same distribution as the reference)
    natural  test images, sent as they are (the test split is sharper than training)
    dark     the same validation images with brightness multiplied by --darken
    blur     the same validation images with a Gaussian blur of radius --blur

The images of a phase are drawn at random with --seed. The clean, dark and blur
phases draw the SAME validation images, so only the transformation differs.
"""

import argparse
import io
import random
import sys
import time
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from src.build_reference import IMAGE_SUFFIXES, load_manifest

PHASES = ("clean", "natural", "dark", "blur")
DEFAULT_URL = "http://localhost:8000/predict"
DEFAULT_MANIFEST = "data/val_manifest.txt"
DEFAULT_TRAIN_DIR = "data/processed/train"
DEFAULT_TEST_DIR = "data/processed/test"
DEFECT_LABEL = "def_front"  # contract with src/inference.py
JPEG_QUALITY = 95


def darken(image: Image.Image, factor: float) -> Image.Image:
    """Multiply the brightness by `factor` (0.5 = half as bright)."""
    return ImageEnhance.Brightness(image).enhance(factor)


def blur(image: Image.Image, radius: float) -> Image.Image:
    """Gaussian blur with the given radius in pixels."""
    return image.filter(ImageFilter.GaussianBlur(radius))


def to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def list_images(directory: str | Path) -> list[Path]:
    """All image files under `directory`, in sorted order."""
    return sorted(
        path
        for path in Path(directory).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validation_paths(train_dir: str | Path, keys: set[str]) -> list[Path]:
    """Paths of the manifest images, which live under the training folder."""
    paths = [Path(train_dir) / key for key in sorted(keys)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(
            f"{len(missing)} manifest images not found under {train_dir}, "
            f"for example {missing[0]}"
        )
    return paths


def pick(paths: list[Path], n: int, rng: random.Random) -> list[Path]:
    """n paths drawn at random (without replacement unless the pool is smaller)."""
    if not paths:
        raise ValueError("no images to draw from")
    if n <= len(paths):
        return rng.sample(paths, n)
    return rng.choices(paths, k=n)


def prepare_payload(
    path: Path, phase: str, darken_factor: float, blur_radius: float
) -> bytes:
    """The bytes to send for one image in a given phase."""
    if phase in ("clean", "natural"):
        return path.read_bytes()
    with Image.open(path) as image:
        image.load()
        if phase == "dark":
            return to_jpeg_bytes(darken(image, darken_factor))
        if phase == "blur":
            return to_jpeg_bytes(blur(image, blur_radius))
    raise ValueError(f"unknown phase: {phase}")


def run_phase(
    phase: str,
    pool: list[Path],
    n: int,
    send,
    rate: float,
    rng: random.Random,
    darken_factor: float = 0.5,
    blur_radius: float = 2.0,
    sleep=time.sleep,
    clock=time.monotonic,
) -> dict:
    """Send n images of one phase at about `rate` requests per second.

    `send` takes image bytes and returns the API response as a dict
    ({"label", "probability"}); it is injected so tests need no server.
    `sleep` and `clock` are injected too, so tests can check the pacing.
    """
    interval = 1.0 / rate
    next_time = clock()
    n_defect = 0
    for path in pick(pool, n, rng):
        result = send(prepare_payload(path, phase, darken_factor, blur_radius))
        if result.get("label") == DEFECT_LABEL:
            n_defect += 1
        next_time += interval
        delay = next_time - clock()
        if delay > 0:
            sleep(delay)
    return {"phase": phase, "sent": n, "defect_rate": n_defect / n}


def make_http_sender(url: str):
    """A send(bytes) -> dict function that posts to the API."""
    import requests

    session = requests.Session()

    def send(payload: bytes) -> dict:
        response = session.post(
            url, files={"file": ("image.jpg", payload, "image/jpeg")}, timeout=30
        )
        response.raise_for_status()
        return response.json()

    return send


def main(argv: list[str] | None = None, sender=None, sleep=time.sleep) -> int:
    parser = argparse.ArgumentParser(description="Replay traffic to show drift")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--phases", default=",".join(PHASES),
        help=f"comma-separated, in order; choose from {', '.join(PHASES)}",
    )
    parser.add_argument("--per-phase", default=300, type=int)
    parser.add_argument("--rate", default=3.0, type=float, help="requests per second")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--darken", default=0.5, type=float, help="brightness factor")
    parser.add_argument("--blur", default=2.0, type=float, help="blur radius in pixels")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    args = parser.parse_args(argv)

    phases = [name.strip() for name in args.phases.split(",") if name.strip()]
    unknown = [name for name in phases if name not in PHASES]
    if not phases or unknown:
        parser.error(f"unknown or missing phases {unknown or ''}; choose from {PHASES}")
    if args.per_phase < 1 or args.rate <= 0 or args.darken <= 0 or args.blur <= 0:
        parser.error("--per-phase, --rate, --darken and --blur must be positive")

    pools: dict[str, list[Path]] = {}
    try:
        if any(name != "natural" for name in phases):
            keys = load_manifest(args.manifest)
            pools["validation"] = validation_paths(args.train_dir, keys)
        if "natural" in phases:
            pools["test"] = list_images(args.test_dir)
            if not pools["test"]:
                raise ValueError(f"no images found in {args.test_dir}")
    except (OSError, ValueError) as exc:
        print(f"Cannot load the images: {exc}", file=sys.stderr)
        return 1

    send = sender or make_http_sender(args.url)

    for phase in phases:
        pool_name = "test" if phase == "natural" else "validation"
        rng = random.Random(f"{args.seed}:{pool_name}")
        print(f"phase {phase}: sending {args.per_phase} images at {args.rate:g}/s ...")
        try:
            summary = run_phase(
                phase, pools[pool_name], args.per_phase, send, args.rate, rng,
                darken_factor=args.darken, blur_radius=args.blur, sleep=sleep,
            )
        except OSError as exc:
            print(
                f"API not reachable at {args.url} ({exc}). "
                "Is the stack running (docker compose up -d)?",
                file=sys.stderr,
            )
            return 1
        print(
            f"phase {phase}: done, {summary['sent']} requests, "
            f"defect rate {summary['defect_rate']:.3f}"
        )

    print(f"Now score the log, for example: python -m src.drift --window {args.per_phase}")
    return 0


if __name__ == "__main__":
    sys.exit(main())