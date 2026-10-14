from common import run_checks


def make_cfg():
    return {
        "baseline": {
            "errors": {
                "missed_defects": 10,
                "false_alarms": 0,
            }
        },
        "policy": {
            "regression_margin_errors": {
                "missed_defects": 2,
                "false_alarms": 5,
            },
            "absolute_floor_errors": {
                "max_missed_defects": 20,
                "max_false_alarms": 3,
            },
        },
    }

def test_candidate_at_limit_passes():
    cfg = make_cfg()
    candidate = {"missed_defects": 12, "false_alarms": 3}
 
    checks = run_checks(candidate, cfg)
 
    assert all(c["passed"] for c in checks)

def test_candidate_over_limit_fails():
    cfg = make_cfg()
    candidate = {"missed_defects": 13, "false_alarms": 4}
 
    checks = run_checks(candidate, cfg)
 
    assert not any(c["passed"] for c in checks)

def test_limits_come_from_the_right_rule():
    cfg = make_cfg()
    candidate = {"missed_defects": 0, "false_alarms": 0}
 
    checks = run_checks(candidate, cfg)
 
    by_name = {c["name"]: c for c in checks}
    assert by_name["missed defects"]["decided_by"] == "regression margin"
    assert by_name["false alarms"]["decided_by"] == "floor"