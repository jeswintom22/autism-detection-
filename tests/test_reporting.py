import json

from asd_training import artifacts, reporting

METRICS = {
    "accuracy": 0.75, "balanced_accuracy": 0.75, "sensitivity": 0.75, "specificity": 0.75,
    "precision": 0.75, "f1": 0.75, "roc_auc": 0.875, "average_precision": 0.8875,
    "confusion_matrix": [[3, 1], [1, 3]], "threshold": 0.5,
    "curves": {"roc": {"fpr": [0, 1], "tpr": [0, 1]}, "pr": {"recall": [0, 1], "precision": [1, 0.5]}},
}


def _write(run_dir, payload):
    artifacts.atomic_json(run_dir / "report_data.json", payload)


def test_report_shows_stored_numbers_and_caveats(tmp_path):
    _write(tmp_path, {
        "status": "complete", "primary_model": "primary",
        "development": {"backbone_comparison": [{"backbone": "efficientnet_b0", "bacc": 0.7}],
                        "ablation_comparison": [{"family": "saliency_only", "bacc": 0.6}]},
        "test": {"models": {"primary": METRICS}, "primary_model": "primary", "target_met": False,
                 "n_patients": 8, "intervals": {"primary": {"accuracy": {"low": 0.4093, "high": 0.9285}}},
                 "subgroups": {}},
        "limitations": ["site dependence"],
    })
    path = reporting.render_report(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "0.7500" in text and "0.8875" in text and "[0.4093, 0.9285]" in text
    assert "NOT MET" in text
    assert "| true control | 3 | 1 |" in text
    assert "saliency" in text and "No clinical variables" in text
    assert "internal, historically explored holdout" in text
    assert "site dependence" in text
    assert (tmp_path / "figures" / "roc.png").is_file()
    assert (tmp_path / "figures" / "pr.png").is_file()


def test_report_without_test_is_marked_incomplete(tmp_path):
    _write(tmp_path, {"status": "development_only", "development": {}, "test": None})
    text = reporting.render_report(tmp_path).read_text(encoding="utf-8")
    assert "incomplete" in text.lower()
    assert "no test evaluation has been recorded" in text
    assert not (tmp_path / "figures").exists()


def test_report_does_not_invent_missing_values(tmp_path):
    _write(tmp_path, {"status": "complete", "test": {"models": {"primary": {"accuracy": 0.9}},
                                                    "primary_model": "primary", "target_met": True}})
    text = reporting.render_report(tmp_path).read_text(encoding="utf-8")
    assert "not available" in text
    assert "MET" in text


def test_report_is_idempotent_and_does_not_modify_inputs(tmp_path):
    _write(tmp_path, {"status": "complete", "test": None})
    before = (tmp_path / "report_data.json").read_bytes()
    first = reporting.render_report(tmp_path).read_text(encoding="utf-8")
    second = reporting.render_report(tmp_path).read_text(encoding="utf-8")
    assert first == second
    assert (tmp_path / "report_data.json").read_bytes() == before
    assert json.loads(before)["status"] == "complete"


def test_report_rejects_unsafe_output_dir_without_creating_it(tmp_path):
    import os
    import pytest

    outside = os.path.abspath(os.sep + "asd_report_outside")
    with pytest.raises(PermissionError):
        reporting.render_report(outside)
    assert not os.path.exists(outside)


def test_report_rejects_unsafe_family_names_before_plotting(tmp_path):
    import pytest

    _write(tmp_path, {"status": "complete", "test": {"models": {"../evil": METRICS}, "primary_model": "../evil"}})
    with pytest.raises(ValueError):
        reporting.render_report(tmp_path)
    assert not (tmp_path / "figures").exists()
    assert not (tmp_path / "report.md").exists()
