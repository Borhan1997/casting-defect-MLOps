import json

import numpy as np
import pytest

from src.drift import (
    InsufficientDataError,
    analyze,
    classify,
    decile_fractions,
    feature_psi,
    format_report,
    load_records,
    main,
    probability_fractions,
    psi,
    reference_probability_fractions,
)

REFERENCE_HISTOGRAM = [0.35, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.07, 0.45]
DECILES = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]


def _make_profile(with_predictions=True):
    rng = np.random.default_rng(0)
    reference = {
        "brightness": rng.normal(140.0, 8.0, 5000),
        "contrast": rng.normal(60.0, 3.5, 5000),
        "sharpness": rng.normal(180.0, 60.0, 5000),
    }
    profile = {
        name: {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "percentiles": np.percentile(values, range(0, 101, 10)).tolist(),
        }
        for name, values in reference.items()
    }
    if with_predictions:
        profile["predictions"] = {
            "n_images": 1000,
            "defect_rate": 0.55,
            "probability": {
                "mean": 0.55,
                "std": 0.45,
                "percentiles": [0.0] * 11,
                "histogram": REFERENCE_HISTOGRAM,
            },
        }
    return profile


def _make_records(
    n,
    *,
    brightness=(140.0, 8.0),
    contrast=(60.0, 3.5),
    sharpness=(180.0, 60.0),
    prob_weights=(0.35, 0.20, 0.45),
    seed=1,
):
    """Synthetic log records. prob_weights = share of probabilities in
    (< 0.1, 0.1 to 0.9, >= 0.9), which matches the reference histogram."""
    rng = np.random.default_rng(seed)
    ranges = [(0.0, 0.099), (0.1, 0.89), (0.9, 1.0)]
    records = []
    for i in range(n):
        low, high = ranges[rng.choice(3, p=prob_weights)]
        probability = float(rng.uniform(low, high))
        records.append(
            {
                "timestamp": f"2026-10-15T08:{i // 60:02d}:{i % 60:02d}+00:00",
                "request_id": f"r{i}",
                "label": "def_front" if probability >= 0.5 else "ok_front",
                "probability": probability,
                "width": 300,
                "height": 300,
                "brightness": float(rng.normal(*brightness)),
                "contrast": float(rng.normal(*contrast)),
                "sharpness": float(rng.normal(*sharpness)),
            }
        )
    return records


def _write_log(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


# --- PSI and binning -------------------------------------------------------


def test_psi_of_identical_distributions_is_zero():
    assert psi([0.25, 0.25, 0.5], [0.25, 0.25, 0.5]) == pytest.approx(0.0)


def test_psi_known_value():
    # 0.4 * ln(0.9 / 0.5) + (-0.4) * ln(0.1 / 0.5)
    assert psi([0.5, 0.5], [0.9, 0.1]) == pytest.approx(0.8789, abs=1e-3)


def test_psi_survives_empty_bins():
    score = psi([0.5, 0.5], [1.0, 0.0])
    assert np.isfinite(score)
    assert score > 0


def test_psi_rejects_different_bin_counts():
    with pytest.raises(ValueError):
        psi([0.5, 0.5], [1.0])


@pytest.mark.parametrize(
    "value, expected",
    [(0.0, "stable"), (0.099, "stable"), (0.1, "moderate"),
     (0.249, "moderate"), (0.25, "major"), (3.0, "major")],
)
def test_classify_thresholds(value, expected):
    assert classify(value) == expected


def test_decile_fractions_clip_out_of_range_values():
    fractions = decile_fractions([-5, 5, 15, 95, 105], DECILES)

    assert len(fractions) == 10
    assert fractions[0] == pytest.approx(0.4)  # -5 and 5
    assert fractions[1] == pytest.approx(0.2)  # 15
    assert fractions[9] == pytest.approx(0.4)  # 95 and 105
    assert sum(fractions) == pytest.approx(1.0)


def test_decile_fractions_reject_non_increasing_edges():
    with pytest.raises(ValueError):
        decile_fractions([1.0, 2.0], [0.0, 5.0, 5.0, 6.0, 10.0])


def test_feature_psi_is_small_for_reference_like_values():
    values = np.random.default_rng(3).normal(140.0, 8.0, 500)
    percentiles = _make_profile()["brightness"]["percentiles"]

    assert feature_psi(values, percentiles) < 0.1


def test_probability_fractions_use_three_bins():
    fractions = probability_fractions([0.05, 0.5, 0.95, 0.99, 0.1, 0.9])

    assert fractions == pytest.approx([1 / 6, 2 / 6, 3 / 6])


def test_reference_probability_fractions_collapse_the_histogram():
    fractions = reference_probability_fractions(REFERENCE_HISTOGRAM)

    assert fractions == pytest.approx([0.35, 0.20, 0.45])


def test_reference_probability_fractions_need_ten_bins():
    with pytest.raises(ValueError):
        reference_probability_fractions([0.5, 0.5])


# --- loading the log -------------------------------------------------------


def test_load_records_keeps_the_last_window_in_order(tmp_path):
    log = tmp_path / "predictions.jsonl"
    _write_log(log, _make_records(10))

    records, n_skipped = load_records(log, window=3)

    assert [r["request_id"] for r in records] == ["r7", "r8", "r9"]
    assert n_skipped == 0


def test_load_records_skips_unusable_lines(tmp_path):
    log = tmp_path / "predictions.jsonl"
    good = _make_records(2)
    log.write_text(
        json.dumps(good[0]) + "\n"
        "\n"                                    # blank: ignored, not counted
        "{not json\n"                           # malformed
        + json.dumps({"label": "ok_front"}) + "\n"           # missing keys
        + json.dumps({**good[1], "brightness": "bright"}) + "\n"  # not numeric
        + json.dumps(good[1]) + "\n"
    )

    records, n_skipped = load_records(log)

    assert [r["request_id"] for r in records] == ["r0", "r1"]
    assert n_skipped == 3


def test_load_records_rejects_a_bad_window(tmp_path):
    log = tmp_path / "predictions.jsonl"
    _write_log(log, _make_records(3))

    with pytest.raises(ValueError):
        load_records(log, window=0)


# --- analysis --------------------------------------------------------------


def test_clean_traffic_is_stable():
    report = analyze(_make_records(500), _make_profile())

    assert report["n_records"] == 500
    assert set(report["signals"]) == {"brightness", "contrast", "sharpness", "probability"}
    for signal in report["signals"].values():
        assert signal["psi"] < 0.1
    assert report["status"] == "stable"


def test_darkened_traffic_is_major_on_all_image_signals():
    records = _make_records(
        500, brightness=(70.0, 4.0), contrast=(30.0, 2.0), sharpness=(45.0, 10.0)
    )

    report = analyze(records, _make_profile())

    for name in ("brightness", "contrast", "sharpness"):
        assert report["signals"][name]["status"] == "major"
    assert report["signals"]["probability"]["status"] == "stable"
    assert report["status"] == "major"


def test_blurred_traffic_moves_sharpness_but_not_brightness():
    report = analyze(_make_records(500, sharpness=(40.0, 10.0)), _make_profile())

    assert report["signals"]["sharpness"]["status"] == "major"
    assert report["signals"]["brightness"]["status"] == "stable"
    assert report["signals"]["contrast"]["status"] == "stable"
    assert report["status"] == "major"


def test_prediction_drift_is_detected_and_defect_rate_is_reported():
    records = _make_records(500, prob_weights=(0.05, 0.90, 0.05))

    report = analyze(records, _make_profile())

    assert report["signals"]["probability"]["status"] == "major"
    assert report["signals"]["brightness"]["status"] == "stable"
    rate = report["defect_rate"]
    assert rate["reference"] == pytest.approx(0.55)
    assert rate["difference"] == pytest.approx(rate["live"] - 0.55)


def test_profile_without_predictions_skips_prediction_signals():
    report = analyze(_make_records(300), _make_profile(with_predictions=False))

    assert set(report["signals"]) == {"brightness", "contrast", "sharpness"}
    assert "defect_rate" not in report


def test_too_few_records_raise():
    with pytest.raises(InsufficientDataError):
        analyze(_make_records(50), _make_profile())


def test_format_report_lists_every_signal_and_the_overall_status():
    report = analyze(_make_records(500), _make_profile())

    text = format_report(report)

    for name in ("brightness", "contrast", "sharpness", "probability"):
        assert name in text
    assert "defect rate" in text
    assert "OVERALL: STABLE" in text


# --- command line ----------------------------------------------------------


def _setup_run(tmp_path, records):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(_make_profile()))
    log = tmp_path / "predictions.jsonl"
    _write_log(log, records)
    return ["--log", str(log), "--profile", str(profile)]


def test_main_returns_0_on_clean_traffic(tmp_path, capsys):
    # 500 records: a clean window of 200 reads "moderate" by chance now and then.
    args = _setup_run(tmp_path, _make_records(500))

    assert main([*args, "--window", "500"]) == 0
    assert "OVERALL: STABLE" in capsys.readouterr().out


def test_main_returns_1_on_major_drift_and_writes_json(tmp_path, capsys):
    args = _setup_run(tmp_path, _make_records(300, sharpness=(40.0, 10.0)))
    report_path = tmp_path / "out" / "report.json"

    assert main([*args, "--json-out", str(report_path)]) == 1

    assert "OVERALL: MAJOR" in capsys.readouterr().out
    assert json.loads(report_path.read_text())["status"] == "major"


def test_main_returns_2_when_the_log_is_missing(tmp_path, capsys):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(_make_profile()))

    code = main(["--log", str(tmp_path / "nope.jsonl"), "--profile", str(profile)])

    assert code == 2
    assert "Cannot compute drift" in capsys.readouterr().err


def test_main_returns_2_when_there_are_too_few_records(tmp_path, capsys):
    args = _setup_run(tmp_path, _make_records(20))

    assert main(args) == 2
    assert "Not enough data" in capsys.readouterr().err


def test_main_respects_min_records(tmp_path):
    args = _setup_run(tmp_path, _make_records(20))

    # Lowering the threshold lets a 20-record window be scored (the result of
    # such a small window is noisy, so only "not refused" is asserted).
    assert main([*args, "--min-records", "10"]) in (0, 1)


def test_main_rejects_a_bad_window(tmp_path):
    args = _setup_run(tmp_path, _make_records(20))

    with pytest.raises(SystemExit):
        main([*args, "--window", "0"])