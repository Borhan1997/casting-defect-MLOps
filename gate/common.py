from pathlib import Path
import json
from pathlib import Path

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