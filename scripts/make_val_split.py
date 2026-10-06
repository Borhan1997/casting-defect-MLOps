"""Carve a fixed, stratified validation set from data/processed/train.

Labels: 1 = def_front (defective), 0 = ok_front (OK).
Writes data/val_manifest.txt (paths relative to data/processed/train/, sorted).

Run from the project root:  python -m src.make_val_split
"""
import hashlib
from pathlib import Path

from sklearn.model_selection import train_test_split

TRAIN_DIR = Path("data/processed/train")
TEST_DIR = Path("data/processed/test")
MANIFEST = Path("data/val_manifest.txt")
CLASSES = {"def_front": 1, "ok_front": 0}
VAL_FRACTION = 0.15
SEED = 42

def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def collect(train_dir: Path) -> tuple[list[str], list[int]]:
    """Return sorted relative paths (e.g. 'def_front/xyz.jpg') and matching labels."""
    paths, labels = [], []
    for cls, label in CLASSES.items():
        for p in sorted((train_dir / cls).glob("*")):
            if p.is_file():
                paths.append(p.relative_to(train_dir).as_posix())
                labels.append(label)
    return paths, labels


def main() -> None:
    paths, labels = collect(TRAIN_DIR)

    _, val_paths, _, val_labels = train_test_split(
        paths,
        labels,
        test_size=VAL_FRACTION,
        stratify=labels,
        random_state=SEED,
    )

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("\n".join(sorted(val_paths)) + "\n")

    # Summary
    print(f"Pool size:        {len(paths)}")
    print(f"Validation size:  {len(val_paths)}")
    print(f"Defective ratio:  pool {sum(labels) / len(labels):.4f} | "
          f"val {sum(val_labels) / len(val_labels):.4f}")

    # Overlap with test, by content hash (filenames can collide across train/test)
    test_hashes = {md5(p) for p in TEST_DIR.rglob("*") if p.is_file()}
    val_hashes = {md5(TRAIN_DIR / p) for p in val_paths}
    overlap = val_hashes & test_hashes
    print(f"Overlap with test (by content): {len(overlap)}")
    assert not overlap, "Validation images found in test/!"

    print(f"Manifest written: {MANIFEST}")


if __name__ == "__main__":
    main()