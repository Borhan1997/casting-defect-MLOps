#!/usr/bin/env python3
"""Split data/processed/train into 3 stratified batches for simulated incremental training.

For each class:
  - list files
  - shuffle independently with a fixed seed (per-class shuffle keeps the split
    exactly stratified, unlike shuffling one combined list)
  - slice at 50% / 75% into batch_1 / batch_2 / batch_3
  - copy into data/processed/batch_N/<class>/

Usage:
    python scripts/make_batches.py
    python scripts/make_batches.py --root data/processed --classes def_front ok_front
"""
import argparse
import random
import shutil
from pathlib import Path

SEED = 42


def list_files(class_dir: Path) -> list[Path]:
    if not class_dir.is_dir():
        raise SystemExit(f"Class directory not found: {class_dir}")
    return sorted(p for p in class_dir.iterdir() if p.is_file())


def split_50_75(files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """Shuffle with a fixed seed and slice at 50% / 75% -> (batch_1, batch_2, batch_3)."""
    shuffled = list(files)
    random.Random(SEED).shuffle(shuffled)
    n = len(shuffled)
    i50, i75 = n // 2, (3 * n) // 4
    return shuffled[:i50], shuffled[i50:i75], shuffled[i75:]


def copy_files(files: list[Path], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in files:
        shutil.copy2(p, dst_dir / p.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/processed"))
    parser.add_argument("--classes", nargs="+", default=["def_front", "ok_front"])
    args = parser.parse_args()

    counts: dict[str, dict[str, int]] = {c: {} for c in args.classes}

    for cls in args.classes:
        files = list_files(args.root / "train" / cls)
        b1, b2, b3 = split_50_75(files)
        for batch_name, batch_files in zip(("batch_1", "batch_2", "batch_3"), (b1, b2, b3)):
            copy_files(batch_files, args.root / batch_name / cls)
            counts[cls][batch_name] = len(batch_files)

    # Summary table
    batch_names = ["batch_1", "batch_2", "batch_3"]
    col_w = max(len(c) for c in args.classes + ["TOTAL"]) + 2
    header = "class".ljust(col_w) + "".join(b.rjust(10) for b in batch_names) + "total".rjust(10)
    print(header)
    print("-" * len(header))
    totals = {b: 0 for b in batch_names}
    for cls in args.classes:
        row_total = sum(counts[cls].values())
        print(
            cls.ljust(col_w)
            + "".join(str(counts[cls][b]).rjust(10) for b in batch_names)
            + str(row_total).rjust(10)
        )
        for b in batch_names:
            totals[b] += counts[cls][b]
    print("-" * len(header))
    grand_total = sum(totals.values())
    print(
        "TOTAL".ljust(col_w)
        + "".join(str(totals[b]).rjust(10) for b in batch_names)
        + str(grand_total).rjust(10)
    )


if __name__ == "__main__":
    main()