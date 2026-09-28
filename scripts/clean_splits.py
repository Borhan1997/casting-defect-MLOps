#!/usr/bin/env python3
"""Build a leakage-free copy of the dataset in data/processed/.

- train is copied unchanged
- test is copied minus every image whose content hash also appears in train
- leaked_files.txt records each removed test image and its matching train file(s)

Usage:
    python scripts/clean_splits.py
    python scripts/clean_splits.py --raw data/raw --out data/processed --overwrite
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

# Reuse the hashing logic (scripts/ is on sys.path when run as `python scripts/clean_splits.py`).
from find_leakage import hash_split


def copy_files(paths: list[Path], src_root: Path, dst_root: Path) -> None:
    """Copy files, preserving their path relative to src_root (keeps class subfolders)."""
    for p in paths:
        dst = dst_root / p.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete the output directory first if it exists")
    args = parser.parse_args()

    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out} already exists. Use --overwrite to rebuild it.")
        shutil.rmtree(args.out)

    raw_train, raw_test = args.raw / "train", args.raw / "test"
    train = hash_split(raw_train)
    test = hash_split(raw_test)

    leaked_hashes = set(train) & set(test)

    train_files = [p for ps in train.values() for p in ps]
    kept_test = [p for h, ps in test.items() if h not in leaked_hashes for p in ps]
    leaked_test = [(h, p) for h in sorted(leaked_hashes) for p in test[h]]

    copy_files(train_files, raw_train, args.out / "train")
    copy_files(kept_test, raw_test, args.out / "test")

    # Audit manifest: one line per removed test image, with its train match(es).
    manifest = args.out / "leaked_files.txt"
    with manifest.open("w") as f:
        f.write("# md5\tremoved_test_file\tmatching_train_file(s)\n")
        for h, p in sorted(leaked_test, key=lambda x: str(x[1])):
            matches = ",".join(str(t) for t in train[h])
            f.write(f"{h}\t{p}\t{matches}\n")

    # Summary, including class breakdown of what was removed (class = parent folder name).
    n_test = sum(len(v) for v in test.values())
    print(f"Train copied: {len(train_files)} images (unchanged)")
    print(f"Test copied:  {len(kept_test)} of {n_test} images")
    print(f"Test removed: {len(leaked_test)} images -> {manifest}")
    removed_by_class = Counter(p.parent.name for _, p in leaked_test)
    kept_by_class = Counter(p.parent.name for p in kept_test)
    print("\nRemoved by class:", dict(removed_by_class))
    print("Remaining test by class:", dict(kept_by_class))


if __name__ == "__main__":
    main()