import json

import numpy as np
import pytest
from PIL import Image

from src.build_reference import FEATURES, build_reference, load_manifest, main


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