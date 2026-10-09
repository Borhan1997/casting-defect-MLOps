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
import sys

import mlflow

from common import (
    BASELINE_PATH,
    EXIT_PASS,
    EXIT_FAIL,
    EXIT_ERROR,
    EvidenceError,
    load_baseline,
    fetch_candidate,
    evaluate_candidate,
    run_checks)



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