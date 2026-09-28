#!/usr/bin/env python3
"""Find exact-duplicate images (by content hash) across and within splits.

Checks:
  1. Test images that also appear in train (train/test leakage).
  2. Duplicate groups inside train (matters for how we build batch/CV splits).

Usage:
    python scripts/find_leakage.py
    python scripts/find_leakage.py --root data/raw --examples 5
"""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def hash_split(split_dir: Path) -> dict[str, list[Path]]:
    """Map content hash -> list of image paths (recursive, so class subfolders work)."""
    hashes: dict[str, list[Path]] = defaultdict(list)
    if not split_dir.is_dir():
        raise SystemExit(f"Split directory not found: {split_dir}")
    for p in sorted(split_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            hashes[md5_of_file(p)].append(p)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--examples", type=int, default=5,
                        help="How many example matches to print per section")
    args = parser.parse_args()

    train = hash_split(args.root / "train")
    test = hash_split(args.root / "test")

    n_train = sum(len(v) for v in train.values())
    n_test = sum(len(v) for v in test.values())
    print(f"Train: {n_train} images ({len(train)} unique hashes)")
    print(f"Test:  {n_test} images ({len(test)} unique hashes)")

    # 1. Train/test overlap
    shared = set(train) & set(test)
    leaked_test_imgs = sum(len(test[h]) for h in shared)
    pct = 100 * leaked_test_imgs / n_test if n_test else 0.0
    print(f"\n=== Train/test leakage ===")
    print(f"{leaked_test_imgs} of {n_test} test images ({pct:.1f}%) "
          f"have an exact match in train ({len(shared)} distinct hashes).")
    for h in sorted(shared)[: args.examples]:
        print(f"  {h}")
        print(f"    test : {[str(p) for p in test[h]]}")
        print(f"    train: {[str(p) for p in train[h]]}")

    # 2. Duplicates within train
    dup_groups = {h: ps for h, ps in train.items() if len(ps) > 1}
    redundant = sum(len(ps) - 1 for ps in dup_groups.values())
    print(f"\n=== Duplicates within train ===")
    print(f"{len(dup_groups)} duplicate groups, {redundant} redundant images.")
    for h, ps in list(dup_groups.items())[: args.examples]:
        print(f"  {h}")
        for p in ps:
            print(f"    {p}")

    # 3. Duplicates within test (cheap to report, useful context)
    test_dups = sum(len(ps) - 1 for ps in test.values() if len(ps) > 1)
    print(f"\n(Duplicates within test: {test_dups} redundant images.)")


if __name__ == "__main__":
    main()