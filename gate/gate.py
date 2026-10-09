"""Evaluation gate: compare a candidate's test-set evaluation run against the baseline.

Usage (from the repo root):
    python gate/gate.py --eval-run-id <EVAL_RUN_ID> [--tracking-uri sqlite:///mlflow.db]

The candidate is identified by the run ID of its *evaluation* run (the one
created by src/evaluate.py), not by its training run.

Exit codes:
    0  PASS   - every check passed
    1  FAIL   - the model is too weak (at least one check failed)
    2  ERROR  - the evidence or the gate itself is broken (nothing was judged)

Rule, expressed in error counts (a lower count is better):
    allowed = min(absolute_floor, baseline_errors + regression_margin)
    candidate passes a check if its error count <= allowed
"""

import argparse
import json
import sys
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"

EXIT_PASS, EXIT_FAIL, EXIT_ERROR = 0, 1, 2

REQUIRED_METRICS = ["test_recall_def", "test_recall_ok", "test_accuracy", "test_loss"]
REQUIRED_PARAMS = ["test_size", "test_defective", "test_ok"]
# Baseline fields that must be filled in before the gate is allowed to run.
REQUIRED_BASELINE_FIELDS = ["training_run_id", "eval_run_id", "data_version", "git_commit"]


class EvidenceError(Exception):
    """The inputs are missing or inconsistent, so no verdict can be given."""


def load_baseline(path: Path) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise EvidenceError(f"baseline file not found: {path}")
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"baseline file is not valid JSON: {exc}")

    missing = [k for k in REQUIRED_BASELINE_FIELDS if not data["baseline"].get(k)]
    if missing:
        raise EvidenceError(f"baseline.json has empty fields: {', '.join(missing)}")
    return data


def fetch_candidate(run_id: str):
    client = MlflowClient()
    try:
        run = client.get_run(run_id)
    except MlflowException as exc:
        raise EvidenceError(f"cannot load run {run_id}: {exc}")

    if run.info.status != "FINISHED":
        raise EvidenceError(f"run {run_id} has status {run.info.status}, expected FINISHED")

    missing_m = [m for m in REQUIRED_METRICS if m not in run.data.metrics]
    missing_p = [p for p in REQUIRED_PARAMS if p not in run.data.params]
    if missing_m or missing_p:
        raise EvidenceError(
            f"run {run_id} is missing metrics {missing_m} / params {missing_p}; "
            "is it an evaluation run created by src/evaluate.py?"
        )
    if "training_run_id" not in run.data.tags:
        raise EvidenceError(f"run {run_id} has no training_run_id tag")
    return run


def recover_errors(recall: float, n: int, label: str) -> int:
    """Turn a recall value back into an error count; verify the round trip."""
    errors = round((1.0 - recall) * n)
    if abs((1.0 - errors / n) - recall) > 1e-9:
        raise EvidenceError(
            f"{label}: recall {recall} is not consistent with {n} images "
            f"(nearest error count {errors})"
        )
    return errors


def evaluate_candidate(run, baseline_cfg: dict) -> dict:
    expected = baseline_cfg["baseline"]["test_set"]
    params = run.data.params
    size, n_def, n_ok = (int(params[k]) for k in REQUIRED_PARAMS)
    if (size, n_def, n_ok) != (expected["size"], expected["defective"], expected["ok"]):
        raise EvidenceError(
            f"test set mismatch: run has {size}/{n_def}/{n_ok}, "
            f"baseline expects {expected['size']}/{expected['defective']}/{expected['ok']}"
        )

    m = run.data.metrics
    return {
        "missed_defects": recover_errors(m["test_recall_def"], n_def, "defect recall"),
        "false_alarms": recover_errors(m["test_recall_ok"], n_ok, "OK recall"),
        "metrics": m,
        "n_def": n_def,
        "n_ok": n_ok,
    }


def run_checks(candidate: dict, cfg: dict) -> list[dict]:
    base = cfg["baseline"]["errors"]
    margin = cfg["policy"]["regression_margin_errors"]
    floor = cfg["policy"]["absolute_floor_errors"]

    specs = [
        ("missed defects", "missed_defects", "missed_defects", "max_missed_defects"),
        ("false alarms", "false_alarms", "false_alarms", "max_false_alarms"),
    ]
    checks = []
    for label, cand_key, base_key, floor_key in specs:
        regression_limit = base[base_key] + margin[base_key]
        allowed = min(floor[floor_key], regression_limit)
        value = candidate[cand_key]
        checks.append(
            {
                "name": label,
                "baseline": base[base_key],
                "candidate": value,
                "allowed": allowed,
                "decided_by": "floor" if floor[floor_key] < regression_limit else "regression margin",
                "passed": value <= allowed,
            }
        )
    return checks


def print_report(checks: list[dict], candidate: dict, cfg: dict, run) -> None:
    base_metrics = cfg["baseline"]["metrics"]
    print(f"Candidate eval run : {run.info.run_id}")
    print(f"Candidate training : {run.data.tags.get('training_run_id')}")
    print(f"Baseline           : {cfg['baseline']['name']} (eval run {cfg['baseline']['eval_run_id']})")
    print()
    header = f"{'check':<16}{'baseline':>10}{'candidate':>11}{'allowed <=':>12}  {'rule':<18}result"
    print(header)
    print("-" * len(header))
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(
            f"{c['name']:<16}{c['baseline']:>10}{c['candidate']:>11}{c['allowed']:>12}  "
            f"{c['decided_by']:<18}{status}"
        )
    print()
    print("Logged only (not gated):")
    for key in cfg["policy"]["logged_only"]:
        print(f"  {key:<14} baseline {base_metrics[key]:.4f}   candidate {candidate['metrics'][key]:.4f}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluation gate")
    parser.add_argument("--eval-run-id", required=True, help="run ID of the candidate's evaluation run")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI; use the same one evaluate.py uses (default: MLflow's own default / env var)",
    )
    args = parser.parse_args()

    try:
        if args.tracking_uri:
            mlflow.set_tracking_uri(args.tracking_uri)
        cfg = load_baseline(BASELINE_PATH)
        run = fetch_candidate(args.eval_run_id)
        candidate = evaluate_candidate(run, cfg)
        checks = run_checks(candidate, cfg)
    except EvidenceError as exc:
        print(f"GATE ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print_report(checks, candidate, cfg, run)
    if all(c["passed"] for c in checks):
        print("GATE: PASS")
        return EXIT_PASS
    print("GATE: FAIL")
    return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())