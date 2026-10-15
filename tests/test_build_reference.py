import json

import numpy as np
import pytest
from PIL import Image

from src.build_reference import (
    FEATURES,
    build_prediction_reference,
    build_reference,
    load_manifest,
    main,
)


def _make_images(root, n=10):
    """Save n noise images split over two class folders; return their keys."""
    keys = []
    for i in range(n):
        class_dir = root / ("def_front" if i % 2 == 0 else "ok_front")
        class_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(i)
        pixels = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
        Image.fromarray(pixels, mode="L").save(class_dir / f"img_{i}.png")
        keys.append(f"{class_dir.name}/img_{i}.png")
    return keys


def test_profile_structure(tmp_path):
    _make_images(tmp_path)

    profile = build_reference(tmp_path)

    assert profile["n_images"] == 10
    assert profile["n_excluded"] == 0
    for name in FEATURES:
        percentiles = profile[name]["percentiles"]
        assert len(percentiles) == 11
        assert percentiles == sorted(percentiles)
    json.dumps(profile)


def test_exclude_skips_listed_images(tmp_path):
    keys = _make_images(tmp_path)

    profile = build_reference(tmp_path, exclude=set(keys[:3]))

    assert profile["n_images"] == 7
    assert profile["n_excluded"] == 3


def test_non_image_files_are_ignored(tmp_path):
    _make_images(tmp_path, n=4)
    (tmp_path / "notes.txt").write_text("not an image")

    assert build_reference(tmp_path)["n_images"] == 4


def test_empty_directory_raises(tmp_path):
    with pytest.raises(ValueError):
        build_reference(tmp_path)


def test_load_manifest_ignores_blank_lines(tmp_path):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("def_front/a.jpg\n\nok_front/b.jpg\n")

    assert load_manifest(manifest) == {"def_front/a.jpg", "ok_front/b.jpg"}


def test_main_writes_json(tmp_path):
    images = tmp_path / "images"
    keys = _make_images(images)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(keys[:2]) + "\n")
    output = tmp_path / "out" / "reference.json"

    main(
        [
            "--images-dir", str(images),
            "--output", str(output),
            "--val-manifest", str(manifest),
        ]
    )

    profile = json.loads(output.read_text())
    assert profile["n_images"] == 8
    assert profile["n_excluded"] == 2
    assert "predictions" not in profile


def _stub_predict(probabilities):
    """Return a fake predict(): yields the given probabilities in call order."""
    results = iter(probabilities)

    def predict(_image_bytes):
        probability = next(results)
        label = "def_front" if probability >= 0.5 else "ok_front"
        return {"label": label, "probability": probability}

    return predict


def test_prediction_reference(tmp_path):
    _make_images(tmp_path)
    # Images are visited in sorted path order:
    # def_front/img_0, def_front/img_2, ok_front/img_1, ok_front/img_3
    include = {
        "def_front/img_0.png",
        "def_front/img_2.png",
        "ok_front/img_1.png",
        "ok_front/img_3.png",
    }

    reference = build_prediction_reference(
        tmp_path, include, _stub_predict([0.9, 0.8, 0.1, 0.2])
    )

    probability = reference["probability"]
    assert reference["n_images"] == 4
    assert reference["defect_rate"] == pytest.approx(0.5)
    assert probability["mean"] == pytest.approx(0.5)
    assert len(probability["percentiles"]) == 11
    assert len(probability["histogram"]) == 10
    assert sum(probability["histogram"]) == pytest.approx(1.0)
    json.dumps(reference)


def test_prediction_histogram_edges(tmp_path):
    _make_images(tmp_path, n=2)
    include = {"def_front/img_0.png", "ok_front/img_1.png"}

    reference = build_prediction_reference(
        tmp_path, include, _stub_predict([1.0, 0.0])
    )

    histogram = reference["probability"]["histogram"]
    assert histogram[0] == pytest.approx(0.5)
    assert histogram[-1] == pytest.approx(0.5)


def test_prediction_reference_raises_when_nothing_matches(tmp_path):
    _make_images(tmp_path, n=2)

    with pytest.raises(ValueError):
        build_prediction_reference(tmp_path, {"nope/missing.png"}, _stub_predict([]))


def test_predictions_flag_requires_manifest(tmp_path):
    _make_images(tmp_path)

    with pytest.raises(SystemExit):
        main(["--images-dir", str(tmp_path), "--predictions"])