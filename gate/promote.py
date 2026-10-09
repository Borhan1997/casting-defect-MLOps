"""Promote a candidate to be the new baseline (a deliberate, manual step).

Usage (from the repo root):
    python gate/promote.py --eval-run-id <EVAL_RUN_ID> [--tracking-uri ...]            # dry run
    python gate/promote.py --eval-run-id <EVAL_RUN_ID> [--tracking-uri ...] --apply    # write file

A candidate is eligible only if:
    1. it passes the gate against the current baseline, and
    2. it has at least `regression_margin_errors.missed_defects` FEWER missed
       defects than the baseline (an improvement larger than the noise margin).

By default nothing is written; --apply is required. Never run this from CI.

Exit codes:
    0  eligible (dry run) or baseline updated (--apply)
    1  refused: candidate is not eligible
    2  ERROR: the evidence or the script is broken
"""

import argparse
import copy
import json
import os
import sys

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from common import (
    BASELINE_PATH,
    EXIT_ERROR,
    EXIT_FAIL,
    EXIT_PASS,
    EvidenceError,
    evaluate_candidate,
    fetch_candidate,
    load_baseline,
    run_checks,
)


def fetch_training_info(eval_run) -> dict:
    """Read the facts the baseline needs from the candidate's *training* run."""
    training_run_id = eval_run.data.tags.get("training_run_id")
    if not training_run_id:
        raise EvidenceError(f"eval run {eval_run.info.run_id} has no training_run_id tag")

    try:
        training_run = MlflowClient().get_run(training_run_id)
    except MlflowException as exc:
        raise EvidenceError(f"cannot load training run {training_run_id}: {exc}")

    data_version = training_run.data.params.get("data_version")
    git_commit = training_run.data.tags.get("git_commit")
    missing = [
        name
        for name, value in (("data_version (param)", data_version), ("git_commit (tag)", git_commit))
        if not value
    ]
    if missing:
        raise EvidenceError(f"training run {training_run_id} is missing: {', '.join(missing)}")

    return {
        "name": training_run.info.run_name or training_run_id,
        "training_run_id": training_run_id,
        "data_version": data_version,
        "git_commit": git_commit,
    }


def eligibility_problems(checks: list[dict], candidate: dict, cfg: dict, eval_run_id: str) -> list[str]:
    """Return the reasons the candidate may NOT be promoted (empty list = eligible)."""
    problems = []
    base = cfg["baseline"]

    if eval_run_id == base["eval_run_id"]:
        problems.append("candidate is the current baseline's own eval run (nothing to promote)")

    failed = [c["name"] for c in checks if not c["passed"]]
    if failed:
        problems.append(f"candidate fails the gate on: {', '.join(failed)}")

    margin = cfg["policy"]["regression_margin_errors"]["missed_defects"]
    required = base["errors"]["missed_defects"] - margin
    if candidate["missed_defects"] > required:
        problems.append(
            f"missed defects {candidate['missed_defects']} are not at least {margin} below the "
            f"baseline's {base['errors']['missed_defects']} (need <= {required})"
        )
    return problems


def build_new_config(cfg: dict, candidate: dict, eval_run, training: dict) -> dict:
    """New baseline block from the candidate; the policy block is copied unchanged."""
    old_metric_keys = cfg["baseline"]["metrics"].keys()
    missing = [k for k in old_metric_keys if k not in candidate["metrics"]]
    if missing:
        raise EvidenceError(f"candidate eval run is missing metrics: {', '.join(missing)}")

    new_cfg = copy.deepcopy(cfg)
    new_cfg["baseline"] = {
        "name": training["name"],
        "training_run_id": training["training_run_id"],
        "eval_run_id": eval_run.info.run_id,
        "data_version": training["data_version"],
        "git_commit": training["git_commit"],
        "test_set": cfg["baseline"]["test_set"],  # already verified equal by evaluate_candidate
        "errors": {
            "missed_defects": candidate["missed_defects"],
            "false_alarms": candidate["false_alarms"],
        },
        "metrics": {k: candidate["metrics"][k] for k in old_metric_keys},
    }
    return new_cfg


def print_comparison(old_cfg: dict, new_cfg: dict) -> None:
    old, new = old_cfg["baseline"], new_cfg["baseline"]
    rows = [
        ("name", old["name"], new["name"]),
        ("data_version", old["data_version"], new["data_version"]),
        ("git_commit", old["git_commit"][:10], new["git_commit"][:10]),
        ("eval_run_id", old["eval_run_id"], new["eval_run_id"]),
        ("missed defects", old["errors"]["missed_defects"], new["errors"]["missed_defects"]),
        ("false alarms", old["errors"]["false_alarms"], new["errors"]["false_alarms"]),
        ("test_accuracy", f"{old['metrics']['test_accuracy']:.4f}", f"{new['metrics']['test_accuracy']:.4f}"),
        ("test_loss", f"{old['metrics']['test_loss']:.4f}", f"{new['metrics']['test_loss']:.4f}"),
    ]
    print(f"{'field':<16}{'current baseline':<36}{'new baseline'}")
    print("-" * 88)
    for field, a, b in rows:
        print(f"{field:<16}{str(a):<36}{b}")
    print()


def write_config(path, cfg: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)  # atomic: never leaves a half-written baseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a candidate to the new baseline")
    parser.add_argument("--eval-run-id", required=True, help="run ID of the candidate's evaluation run")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI (same as evaluate.py)")
    parser.add_argument("--apply", action="store_true", help="actually write gate/baseline.json")
    args = parser.parse_args()

    try:
        if args.tracking_uri:
            mlflow.set_tracking_uri(args.tracking_uri)
        cfg = load_baseline(BASELINE_PATH)
        run = fetch_candidate(args.eval_run_id)
        candidate = evaluate_candidate(run, cfg)
        checks = run_checks(candidate, cfg)
        problems = eligibility_problems(checks, candidate, cfg, args.eval_run_id)
        if problems:
            print("PROMOTION REFUSED:")
            for p in problems:
                print(f"  - {p}")
            return EXIT_FAIL
        training = fetch_training_info(run)
        new_cfg = build_new_config(cfg, candidate, run, training)
    except EvidenceError as exc:
        print(f"PROMOTE ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print_comparison(cfg, new_cfg)
    old_e, new_e = cfg["baseline"]["errors"], new_cfg["baseline"]["errors"]
    message = (
        f"chore(baseline): promote {new_cfg['baseline']['name']}, "
        f"misses {old_e['missed_defects']} -> {new_e['missed_defects']}, "
        f"false alarms {old_e['false_alarms']} -> {new_e['false_alarms']} "
        f"(eval run {args.eval_run_id})"
    )

    if not args.apply:
        print("DRY RUN: candidate is eligible, nothing was written. Re-run with --apply to update the baseline.")
        return EXIT_PASS

    write_config(BASELINE_PATH, new_cfg)
    print(f"Baseline updated: {BASELINE_PATH}")
    print("Review with `git diff gate/baseline.json`, then commit on its own:")
    print(f'  git commit -am "{message}"')
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())