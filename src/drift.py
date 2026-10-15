"""Offline drift detection: compare recent logged predictions with the reference profile.

Not part of the serving image (the Dockerfile copies only the files it names).

Usage (from the project root):
    python -m src.drift --log logs/predictions.jsonl --window 200

Signals, all scored with the Population Stability Index (PSI):
- brightness, contrast, sharpness: 10 bins whose edges are the reference deciles,
  so the expected share is 10% per bin. Live values outside the reference range
  fall into the first or the last bin.
- probability: 3 bins (< 0.1, 0.1 to 0.9, >= 0.9) built from the reference histogram.
The defect rate is reported next to its reference but is not scored.

Exit codes: 0 = no major drift, 1 = major drift (PSI >= 0.25),
2 = not enough data or unreadable input.
"""

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

FEATURES = ("brightness", "contrast", "sharpness")
REQUIRED_KEYS = frozenset({"label", "probability", *FEATURES})
DEFECT_LABEL = "def_front"  # contract with src/inference.py

EPSILON = 1e-4  # floor for every bin fraction, so an empty bin cannot divide by zero
PSI_MODERATE = 0.1  # conventional rule of thumb: < 0.1 stable
PSI_MAJOR = 0.25  # 0.1 to 0.25 moderate, 0.25 and above major
SEVERITY = {"stable": 0, "moderate": 1, "major": 2}

DEFAULT_WINDOW = 200
# Simulated on clean data: with 100 records some signal reads "moderate" about
# 3 windows out of 4 by chance; with 200 it happens about 1 window in 13 and
# "major" essentially never. Hence no score below 200 records.
MIN_RECORDS = 200

LOW_PROBABILITY = 0.1
HIGH_PROBABILITY = 0.9

DEFAULT_LOG = "logs/predictions.jsonl"
DEFAULT_PROFILE = "monitoring/reference_profile.json"


class InsufficientDataError(ValueError):
    """Not enough logged records to score drift reliably."""


def psi(expected, actual) -> float:
    """Population Stability Index between two lists of bin fractions."""
    expected = np.maximum(np.asarray(expected, dtype=np.float64), EPSILON)
    actual = np.maximum(np.asarray(actual, dtype=np.float64), EPSILON)
    if expected.shape != actual.shape:
        raise ValueError("expected and actual must have the same number of bins")
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def classify(psi_value: float) -> str:
    """Map a PSI value to 'stable', 'moderate' or 'major'."""
    if psi_value >= PSI_MAJOR:
        return "major"
    if psi_value >= PSI_MODERATE:
        return "moderate"
    return "stable"


def decile_fractions(values, percentiles) -> list[float]:
    """Share of `values` in each bin delimited by the reference percentiles.

    The 11 reference percentiles give 10 bins. Values below the first edge or
    above the last one are counted in the first or the last bin.
    """
    edges = np.asarray(percentiles, dtype=np.float64)
    interior = edges[1:-1]
    if len(interior) < 1 or np.any(np.diff(interior) <= 0):
        raise ValueError("reference percentiles must be strictly increasing")
    n_bins = len(edges) - 1
    index = np.searchsorted(interior, np.asarray(values, dtype=np.float64), side="right")
    counts = np.bincount(index, minlength=n_bins)
    return (counts / len(values)).tolist()


def feature_psi(values, percentiles) -> float:
    """PSI of live values against the reference deciles (10% expected per bin)."""
    n_bins = len(percentiles) - 1
    return psi([1.0 / n_bins] * n_bins, decile_fractions(values, percentiles))


def probability_fractions(probabilities) -> list[float]:
    """Share of probabilities in the three bins: < 0.1, 0.1 to 0.9, >= 0.9."""
    array = np.asarray(probabilities, dtype=np.float64)
    low = int(np.count_nonzero(array < LOW_PROBABILITY))
    high = int(np.count_nonzero(array >= HIGH_PROBABILITY))
    middle = len(array) - low - high
    return [low / len(array), middle / len(array), high / len(array)]


def reference_probability_fractions(histogram) -> list[float]:
    """Collapse the reference 10-bin histogram into the same three bins."""
    if len(histogram) != 10:
        raise ValueError("the reference probability histogram must have 10 bins")
    return [float(histogram[0]), float(sum(histogram[1:9])), float(histogram[9])]


def _is_valid(record) -> bool:
    if not isinstance(record, dict) or not REQUIRED_KEYS.issubset(record):
        return False
    try:
        return all(math.isfinite(float(record[key])) for key in ("probability", *FEATURES))
    except (TypeError, ValueError):
        return False


def load_records(path: str | Path, window: int = DEFAULT_WINDOW) -> tuple[list[dict], int]:
    """Read the last `window` valid records of a JSONL log.

    Returns (records, number of lines skipped because they were malformed).
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    records: deque[dict] = deque(maxlen=window)
    n_skipped = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                n_skipped += 1
                continue
            if _is_valid(record):
                records.append(record)
            else:
                n_skipped += 1
    return list(records), n_skipped


def _signal(score: float, live_mean: float, reference_mean: float) -> dict:
    return {
        "psi": score,
        "status": classify(score),
        "live_mean": live_mean,
        "reference_mean": reference_mean,
    }


def analyze(records: list[dict], profile: dict, min_records: int = MIN_RECORDS) -> dict:
    """Score drift of `records` against the reference `profile`."""
    n = len(records)
    if n < min_records:
        raise InsufficientDataError(
            f"{n} records in the window, at least {min_records} needed"
        )

    signals: dict[str, dict] = {}
    for name in FEATURES:
        values = [float(r[name]) for r in records]
        score = feature_psi(values, profile[name]["percentiles"])
        signals[name] = _signal(
            score, float(np.mean(values)), float(profile[name]["mean"])
        )

    report: dict = {
        "n_records": n,
        "window_start": records[0].get("timestamp"),
        "window_end": records[-1].get("timestamp"),
        "signals": signals,
    }

    predictions = profile.get("predictions")
    if predictions:
        probabilities = [float(r["probability"]) for r in records]
        expected = reference_probability_fractions(predictions["probability"]["histogram"])
        score = psi(expected, probability_fractions(probabilities))
        signals["probability"] = _signal(
            score,
            float(np.mean(probabilities)),
            float(predictions["probability"]["mean"]),
        )
        live_rate = sum(1 for r in records if r["label"] == DEFECT_LABEL) / n
        reference_rate = float(predictions["defect_rate"])
        report["defect_rate"] = {
            "live": live_rate,
            "reference": reference_rate,
            "difference": live_rate - reference_rate,
        }

    report["status"] = max(
        (signal["status"] for signal in signals.values()), key=SEVERITY.__getitem__
    )
    return report


def format_report(report: dict) -> str:
    """Human-readable table for the terminal."""
    lines = [
        (
            f"Drift report: {report['n_records']} records, "
            f"{report['window_start']} to {report['window_end']}"
        ),
        "",
        f"{'signal':<12}{'psi':>8}  {'status':<10}{'live mean':>12}{'reference':>12}",
        "-" * 54,
    ]
    for name, signal in report["signals"].items():
        lines.append(
            f"{name:<12}{signal['psi']:>8.3f}  {signal['status']:<10}"
            f"{signal['live_mean']:>12.3f}{signal['reference_mean']:>12.3f}"
        )
    rate = report.get("defect_rate")
    if rate:
        lines += [
            "",
            (
                f"defect rate: live {rate['live']:.3f} vs reference "
                f"{rate['reference']:.3f} (difference {rate['difference']:+.3f})"
            ),
        ]
    lines += ["", f"OVERALL: {report['status'].upper()}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score drift of logged predictions against the reference profile"
    )
    parser.add_argument("--log", default=DEFAULT_LOG, type=Path)
    parser.add_argument("--profile", default=DEFAULT_PROFILE, type=Path)
    parser.add_argument(
        "--window", default=DEFAULT_WINDOW, type=int,
        help="number of most recent records to score",
    )
    parser.add_argument(
        "--min-records", default=MIN_RECORDS, type=int,
        help="refuse to score fewer records than this",
    )
    parser.add_argument(
        "--json-out", default=None, type=Path,
        help="also write the report as JSON to this path",
    )
    args = parser.parse_args(argv)

    if args.window < 1:
        parser.error("--window must be at least 1")

    try:
        profile = json.loads(args.profile.read_text())
        records, n_skipped = load_records(args.log, args.window)
        report = analyze(records, profile, args.min_records)
    except InsufficientDataError as exc:
        print(f"Not enough data: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Cannot compute drift: {exc!r}", file=sys.stderr)
        return 2

    if n_skipped:
        print(f"Warning: skipped {n_skipped} unreadable log lines", file=sys.stderr)

    print(format_report(report))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")

    return 1 if report["status"] == "major" else 0


if __name__ == "__main__":
    sys.exit(main())