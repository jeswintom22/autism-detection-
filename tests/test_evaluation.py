import json

import numpy as np
import pytest

from asd_training import evaluation

Y = [0, 0, 0, 0, 1, 1, 1, 1]
P = [0.1, 0.2, 0.7, 0.3, 0.9, 0.8, 0.4, 0.6]


def test_known_answer_metrics():
    m = evaluation.patient_metrics(Y, P, 0.5)
    assert m["confusion_matrix"] == [[3, 1], [1, 3]]
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 3)
    for key in ("accuracy", "balanced_accuracy", "precision", "recall", "sensitivity", "specificity", "f1"):
        assert m[key] == pytest.approx(0.75)
    assert m["roc_auc"] == pytest.approx(0.875)
    assert m["average_precision"] == pytest.approx(0.8875)
    assert m["n_patients"] == 8 and m["n_asd"] == 4 and m["n_control"] == 4
    assert m["threshold"] == 0.5 and m["warnings"] == []


def test_wilson_interval_known_values():
    low, high = evaluation.wilson_interval(6, 8)
    assert low == pytest.approx(0.4092754303, abs=1e-9)
    assert high == pytest.approx(0.9285207872, abs=1e-9)
    assert evaluation.wilson_interval(0, 0) == [None, None]
    assert evaluation.wilson_interval(0, 10)[0] == 0.0 and evaluation.wilson_interval(10, 10)[1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        evaluation.wilson_interval(9, 8)


def test_threshold_boundary_is_greater_or_equal():
    below = evaluation.patient_metrics([0, 1], [0.4, 0.6], 0.6)
    assert below["tp"] == 1 and below["fp"] == 0
    above = evaluation.patient_metrics([0, 1], [0.4, 0.6], 0.6000001)
    assert above["tp"] == 0 and above["fn"] == 1


def test_threshold_must_be_valid_and_probabilities_bounded():
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            evaluation.patient_metrics(Y, P, bad)
    with pytest.raises(ValueError):
        evaluation.patient_metrics(Y, [1.2] + P[1:], 0.5)
    with pytest.raises(ValueError):
        evaluation.patient_metrics([0, 2], [0.1, 0.2], 0.5)
    with pytest.raises(ValueError):
        evaluation.patient_metrics([0, 1], [0.1], 0.5)


def test_no_positive_predictions_yield_zero_precision_with_warning():
    m = evaluation.patient_metrics(Y, [0.1] * 8, 0.5)
    assert m["precision"] == 0.0 and m["f1"] == 0.0 and m["sensitivity"] == 0.0
    assert any("No positive predictions" in w for w in m["warnings"])


def test_one_class_metrics_are_null_not_fabricated():
    m = evaluation.patient_metrics([1, 1, 1], [0.9, 0.2, 0.7], 0.5)
    assert m["specificity"] is None and m["balanced_accuracy"] is None
    assert m["roc_auc"] is None and m["average_precision"] is None
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["accuracy"] == pytest.approx(2 / 3)
    assert m["n_control"] == 0 and m["warnings"]
    json.dumps(m, allow_nan=False)
    empty = evaluation.patient_metrics([], [], 0.5)
    assert empty["accuracy"] is None and empty["n_patients"] == 0
    json.dumps(empty, allow_nan=False)


def test_best_threshold_grid_ties_and_range():
    assert evaluation.best_threshold([0, 1], [0.2, 0.8]) == 0.5
    assert evaluation.best_threshold([0, 1], [0.3, 0.4]) == 0.4
    assert evaluation.best_threshold([0, 1], [0.6, 0.7]) == 0.61
    assert evaluation.best_threshold([0, 1], [0.6, 0.9]) == 0.61
    assert evaluation.best_threshold([0, 1], [0.01, 0.99]) == 0.5
    assert evaluation.best_threshold([0, 1, 0, 1], [0.97, 0.98, 0.96, 0.99]) == 0.5
    with pytest.raises(ValueError):
        evaluation.best_threshold([1, 1], [0.5, 0.6])


def test_best_threshold_equidistant_tie_prefers_lower():
    y = [0, 0, 1, 1]
    p = [0.1, 0.45, 0.55, 0.9]
    assert evaluation.best_threshold(y, p) == 0.5
    y2 = [0, 1, 0, 1, 1, 0]
    p2 = [0.3, 0.7, 0.3, 0.7, 0.3, 0.7]
    t = evaluation.best_threshold(y2, p2)
    scores = {c / 100: evaluation.patient_metrics(y2, p2, c / 100)["balanced_accuracy"] for c in range(5, 96)}
    best = max(scores.values())
    tied = [c for c, v in scores.items() if abs(v - best) < 1e-12]
    assert t == min(tied, key=lambda c: (round(abs(c - 0.5), 12), c))


def test_bootstrap_is_seeded_stratified_and_reports_metadata():
    rng = np.random.default_rng(3)
    y = np.array([0] * 30 + [1] * 30)
    p = np.clip(0.4 * y + 0.3 + rng.normal(0, 0.2, 60), 0, 1)
    a = evaluation.bootstrap_intervals(y, p, 0.5, n_bootstrap=50, seed=7)
    b = evaluation.bootstrap_intervals(y, p, 0.5, n_bootstrap=50, seed=7)
    c = evaluation.bootstrap_intervals(y, p, 0.5, n_bootstrap=50, seed=8)
    assert a == b and a != c
    assert a["bootstrap_seed"] == 7 and a["bootstrap_samples"] == 50
    measured = evaluation.patient_metrics(y, p, 0.5)
    for key in ("balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc", "average_precision"):
        assert a[key]["lower"] <= a[key]["upper"]
    assert a["accuracy"]["method"] == "95% Wilson"
    low, high = evaluation.wilson_interval(measured["tp"] + measured["tn"], 60)
    assert a["accuracy"]["lower"] == pytest.approx(low) and a["accuracy"]["upper"] == pytest.approx(high)
    json.dumps(a, allow_nan=False)


def test_bootstrap_requires_both_classes_and_positive_count():
    with pytest.raises(ValueError):
        evaluation.bootstrap_intervals([1, 1, 1], [0.5, 0.6, 0.7])
    with pytest.raises(ValueError):
        evaluation.bootstrap_intervals(Y, P, n_bootstrap=0)


def test_stored_curves_are_plottable_coordinates():
    curves = evaluation.stored_curves(Y, P)
    assert curves["roc"]["fpr"][0] == 0 and curves["roc"]["tpr"][-1] == 1
    assert len(curves["roc"]["fpr"]) == len(curves["roc"]["tpr"])
    assert len(curves["pr"]["recall"]) == len(curves["pr"]["precision"])
    assert evaluation.stored_curves([1, 1], [0.2, 0.3]) == {}
    json.dumps(curves, allow_nan=False)
