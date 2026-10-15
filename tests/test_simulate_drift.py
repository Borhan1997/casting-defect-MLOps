import io
import random

import numpy as np
import pytest
from PIL import Image

from src.monitoring import compute_image_stats
from src.simulate_drift import (
    PHASES,
    blur,
    darken,
    list_images,
    main,
    pick,
    prepare_payload,
    run_phase,
    to_jpeg_bytes,
    validation_paths,
)


def _noise_image(seed=0, size=64):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(40, 216, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _make_images(root, n, prefix="img"):
    """Save n noise PNGs over two class folders; return their manifest-style keys."""
    keys = []
    for i in range(n):
        class_dir = root / ("def_front" if i % 2 == 0 else "ok_front")
        class_dir.mkdir(parents=True, exist_ok=True)
        _noise_image(seed=i).save(class_dir / f"{prefix}_{i}.png")
        keys.append(f"{class_dir.name}/{prefix}_{i}.png")
    return keys


def _stub_send(sent):
    def send(payload):
        sent.append(payload)
        return {"label": "def_front" if len(sent) % 2 else "ok_front", "probability": 0.5}

    return send


# --- transformations -------------------------------------------------------


def test_darken_lowers_brightness():
    image = _noise_image()

    darker = compute_image_stats(darken(image, 0.5))["brightness"]

    assert darker < compute_image_stats(image)["brightness"] * 0.6


def test_blur_lowers_sharpness_and_keeps_brightness():
    image = _noise_image()
    before = compute_image_stats(image)

    after = compute_image_stats(blur(image, 2.0))

    assert after["sharpness"] < before["sharpness"] * 0.5
    assert after["brightness"] == pytest.approx(before["brightness"], abs=3.0)


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_to_jpeg_bytes_decodes_as_jpeg(mode):
    payload = to_jpeg_bytes(_noise_image().convert(mode))

    assert Image.open(io.BytesIO(payload)).format == "JPEG"


# --- selecting images ------------------------------------------------------


def test_list_images_finds_images_recursively_in_sorted_order(tmp_path):
    _make_images(tmp_path, 4)
    (tmp_path / "notes.txt").write_text("not an image")

    paths = list_images(tmp_path)

    assert len(paths) == 4
    assert paths == sorted(paths)


def test_validation_paths_build_from_manifest_keys(tmp_path):
    keys = _make_images(tmp_path, 4)

    paths = validation_paths(tmp_path, set(keys[:3]))

    assert len(paths) == 3
    assert all(path.is_file() for path in paths)


def test_validation_paths_report_missing_images(tmp_path):
    _make_images(tmp_path, 2)

    with pytest.raises(ValueError, match="not found"):
        validation_paths(tmp_path, {"def_front/does_not_exist.png"})


def test_pick_is_reproducible_for_a_seed(tmp_path):
    _make_images(tmp_path, 10)
    paths = list_images(tmp_path)

    first = pick(paths, 5, random.Random("0:validation"))
    second = pick(paths, 5, random.Random("0:validation"))
    other = pick(paths, 5, random.Random("1:validation"))

    assert first == second
    assert len(set(first)) == 5
    assert first != other


def test_pick_repeats_images_when_the_pool_is_smaller(tmp_path):
    _make_images(tmp_path, 3)

    assert len(pick(list_images(tmp_path), 8, random.Random(0))) == 8


def test_pick_rejects_an_empty_pool():
    with pytest.raises(ValueError):
        pick([], 3, random.Random(0))


# --- payloads and phases ---------------------------------------------------


def test_clean_and_natural_payloads_are_the_original_bytes(tmp_path):
    _make_images(tmp_path, 1)
    path = list_images(tmp_path)[0]

    assert prepare_payload(path, "clean", 0.5, 2.0) == path.read_bytes()
    assert prepare_payload(path, "natural", 0.5, 2.0) == path.read_bytes()


def test_dark_payload_is_darker_and_blur_payload_is_blurrier(tmp_path):
    _make_images(tmp_path, 1)
    path = list_images(tmp_path)[0]
    original = compute_image_stats(Image.open(path))

    dark = compute_image_stats(Image.open(io.BytesIO(prepare_payload(path, "dark", 0.5, 2.0))))
    blurred = compute_image_stats(Image.open(io.BytesIO(prepare_payload(path, "blur", 0.5, 2.0))))

    assert dark["brightness"] < original["brightness"] * 0.6
    assert blurred["sharpness"] < original["sharpness"] * 0.5


def test_prepare_payload_rejects_an_unknown_phase(tmp_path):
    _make_images(tmp_path, 1)

    with pytest.raises(ValueError):
        prepare_payload(list_images(tmp_path)[0], "sepia", 0.5, 2.0)


class _FakeClock:
    """A clock that only moves when sleep() is called (sends take no time)."""

    def __init__(self):
        self.now = 0.0
        self.delays = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.delays.append(seconds)
        self.now += seconds


def test_run_phase_sends_n_requests_at_the_requested_pace(tmp_path):
    _make_images(tmp_path, 6)
    sent, clock = [], _FakeClock()

    summary = run_phase(
        "clean", list_images(tmp_path), 10, _stub_send(sent), rate=100.0,
        rng=random.Random(0), sleep=clock.sleep, clock=clock,
    )

    assert len(sent) == 10
    assert summary == {"phase": "clean", "sent": 10, "defect_rate": pytest.approx(0.5)}
    assert clock.delays == pytest.approx([0.01] * 10)  # one interval per request


# --- command line ----------------------------------------------------------


def _setup_dirs(tmp_path):
    train = tmp_path / "train"
    test = tmp_path / "test"
    keys = _make_images(train, 8)
    _make_images(test, 5, prefix="t")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(keys[:6]) + "\n")
    return [
        "--manifest", str(manifest),
        "--train-dir", str(train),
        "--test-dir", str(test),
    ]


def test_main_runs_all_phases(tmp_path, capsys):
    args = _setup_dirs(tmp_path)
    sent = []

    code = main(
        [*args, "--per-phase", "5", "--rate", "1000"],
        sender=_stub_send(sent), sleep=lambda _seconds: None,
    )

    assert code == 0
    assert len(sent) == 5 * len(PHASES)
    out = capsys.readouterr().out
    for phase in PHASES:
        assert f"phase {phase}: done" in out
    assert "src.drift" in out


def test_main_runs_only_the_requested_phases(tmp_path):
    args = _setup_dirs(tmp_path)
    sent = []

    code = main(
        [*args, "--phases", "dark", "--per-phase", "4", "--rate", "1000"],
        sender=_stub_send(sent), sleep=lambda _seconds: None,
    )

    assert code == 0
    assert len(sent) == 4


def test_main_rejects_unknown_phases(tmp_path):
    args = _setup_dirs(tmp_path)

    with pytest.raises(SystemExit):
        main([*args, "--phases", "clean,sepia"], sender=_stub_send([]))


def test_main_rejects_non_positive_options(tmp_path):
    args = _setup_dirs(tmp_path)

    with pytest.raises(SystemExit):
        main([*args, "--rate", "0"], sender=_stub_send([]))


def test_main_returns_1_when_the_api_is_unreachable(tmp_path, capsys):
    args = _setup_dirs(tmp_path)

    def unreachable(_payload):
        raise ConnectionError("connection refused")

    code = main(
        [*args, "--phases", "clean", "--per-phase", "3"],
        sender=unreachable, sleep=lambda _seconds: None,
    )

    assert code == 1
    assert "not reachable" in capsys.readouterr().err


def test_main_returns_1_when_the_manifest_images_are_missing(tmp_path, capsys):
    args = _setup_dirs(tmp_path)
    (tmp_path / "manifest.txt").write_text("def_front/missing.png\n")

    code = main([*args, "--phases", "clean"], sender=_stub_send([]))

    assert code == 1
    assert "Cannot load the images" in capsys.readouterr().err