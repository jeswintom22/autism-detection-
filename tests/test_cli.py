import types

import pytest

from asd_training import artifacts, cli, config as config_module
from asd_training.config import Settings


def _rows():
    rows = []
    n = 0
    for site in ("NYU", "UCLA"):
        for label in (0, 1):
            for _ in range(14):
                n += 1
                rows.append({"subject_id": str(1000 + n), "label": label, "class_name": "asd" if label else "control"})
    return rows


class FakeCore:
    def __init__(self, tmp_path, blocking=None):
        self.calls = []
        self.blocking = blocking or []
        self.source_drift = False
        self.settings = Settings(
            repo_root=tmp_path / "repo",
            control_dir=tmp_path / "repo" / "control",
            treatment_dir=tmp_path / "repo" / "treatment",
        )
        self.config = types.SimpleNamespace(
            load_config=lambda path=None: self.settings,
            settings_from_dict=config_module.settings_from_dict,
            config_hash=config_module.config_hash,
        )
        self.settings.repo_root.mkdir(parents=True, exist_ok=True)
        self.verified = []
        self.dataset = types.SimpleNamespace(audit_dataset=self._audit, verify_sources=self._verify_sources)
        self.splits = types.SimpleNamespace(
            create_splits=self._create_splits,
            create_development_folds=self._dev_folds,
            validate_splits=lambda rows, splits: None,
        )
        self.training = types.SimpleNamespace(train_development=self._train)
        self.evaluation = types.SimpleNamespace(evaluate_frozen=self._evaluate)

    def _verify_sources(self, rows):
        self.verified.append(len(rows))
        if self.source_drift:
            raise ValueError("Source changed since audit for patient 1")

    def _audit(self, settings):
        self.calls.append("audit")
        return _rows(), {"blocking_errors": list(self.blocking), "failures": []}

    def _create_splits(self, rows, settings):
        ids = sorted(r["subject_id"] for r in rows)
        test = ids[::5]
        dev = [i for i in ids if i not in test]
        payload = {"purpose": settings.purpose, "cohort_hash": artifacts.stable_hash(ids),
                   "development_ids": dev, "test_ids": test}
        payload["split_hash"] = artifacts.stable_hash(payload)
        return payload

    def _dev_folds(self, rows, settings):
        ids = sorted(r["subject_id"] for r in rows)
        payload = {"purpose": settings.purpose, "cohort_hash": artifacts.stable_hash(ids),
                   "development_ids": ids, "test_ids": []}
        payload["split_hash"] = artifacts.stable_hash(payload)
        return payload

    def _train(self, run_dir, settings, rows, splits):
        self.calls.append(("train", settings.purpose, [r["subject_id"] for r in rows], splits))
        return {}

    def _evaluate(self, run_dir, settings, rows, splits):
        self.calls.append(("evaluate", [r["subject_id"] for r in rows]))
        return {}


@pytest.fixture
def core(tmp_path, monkeypatch):
    fake = FakeCore(tmp_path)
    (fake.settings.repo_root / "runs" / "hybrid").mkdir(parents=True)
    monkeypatch.setattr(cli, "_core", lambda name: getattr(fake, name))
    return fake


def _run_dir(core, name="r1"):
    return core.settings.repo_root / "runs" / "hybrid" / name


def test_parser_rejects_slice_split_and_legacy_flags():
    for argv in (["train", "--run-dir", "x", "--split", "slice"], ["run", "--run-dir", "x", "--split", "group"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
    with pytest.raises(SystemExit):
        cli.main(["train", "--run-dir", "x", "--epochs", "1"])


def test_evaluate_requires_confirm_flag(core):
    assert cli.main(["evaluate-test", "--run-dir", str(_run_dir(core))]) == 2
    assert "evaluation" not in core.calls


def test_prepare_writes_artifacts_and_separates_partitions(core):
    run_dir = _run_dir(core)
    assert cli.main(["prepare", "--run-dir", str(run_dir)]) == 0
    for name in ("config.json", "environment.json", "audit.json", "source_manifest.csv", "development_manifest.json",
                 "development_manifest.csv", "test_manifest.json", "test_manifest.csv", "splits.json", "run_state.json"):
        assert (run_dir / name).is_file(), name
    splits = artifacts.read_json(run_dir / "splits.json")
    dev = {r["subject_id"] for r in artifacts.read_json(run_dir / "development_manifest.json")["rows"]}
    test = {r["subject_id"] for r in artifacts.read_json(run_dir / "test_manifest.json")["rows"]}
    assert dev == set(splits["development_ids"]) and test == set(splits["test_ids"])
    assert not dev & test
    registry = core.settings.repo_root / "runs" / "hybrid" / "protocols" / splits["cohort_hash"] / "splits.json"
    assert artifacts.read_json(registry) == splits


def test_prepare_is_idempotent_and_refuses_changed_config(core):
    run_dir = _run_dir(core)
    assert cli.main(["prepare", "--run-dir", str(run_dir)]) == 0
    assert cli.main(["prepare", "--run-dir", str(run_dir)]) == 0
    assert core.calls.count("audit") == 1
    core.settings = core.settings.__class__(**{**core.settings.__dict__, "seed": 7})
    assert cli.main(["prepare", "--run-dir", str(run_dir)]) == 2


def test_prepare_refuses_mismatching_registry(core):
    run_dir = _run_dir(core)
    splits = core._create_splits(_rows(), core.settings)
    registry = core.settings.repo_root / "runs" / "hybrid" / "protocols" / splits["cohort_hash"] / "splits.json"
    artifacts.atomic_json(registry, {**splits, "test_ids": [], "split_hash": "other"})
    assert cli.main(["prepare", "--run-dir", str(run_dir)]) == 2
    assert not (run_dir / "splits.json").exists()


def test_prepare_halts_on_blocking_audit_errors(core):
    core.blocking = ["duplicate voxel content"]
    run_dir = _run_dir(core)
    assert cli.main(["prepare", "--run-dir", str(run_dir)]) == 2
    assert (run_dir / "audit.json").is_file()
    assert not (run_dir / "splits.json").exists()
    assert not (core.settings.repo_root / "runs" / "hybrid" / "protocols").exists()


def test_prepare_rejects_protected_output(core):
    assert cli.main(["prepare", "--run-dir", str(core.settings.repo_root / "results_new")]) == 2
    assert "audit" not in core.calls


def test_train_never_loads_test_manifest(core, monkeypatch):
    run_dir = _run_dir(core)
    cli.main(["prepare", "--run-dir", str(run_dir)])

    def boom(*args, **kwargs):
        raise AssertionError("test manifest loaded during training")

    monkeypatch.setattr(cli, "_load_test_manifest", boom)
    (run_dir / "test_manifest.json").unlink()
    assert cli.main(["train", "--run-dir", str(run_dir)]) == 0
    _, purpose, ids, splits = core.calls[-1]
    assert purpose == "production"
    assert set(ids) == set(splits["development_ids"])
    assert not set(ids) & set(splits["test_ids"])


def test_train_detects_tampered_development_manifest(core):
    run_dir = _run_dir(core)
    cli.main(["prepare", "--run-dir", str(run_dir)])
    rows = artifacts.read_json(run_dir / "development_manifest.json")["rows"]
    artifacts.atomic_json(run_dir / "development_manifest.json", {"rows": rows[:-1]})
    assert cli.main(["train", "--run-dir", str(run_dir)]) == 2


def test_evaluate_test_requires_frozen_bundle_then_scores_test_rows(core, tmp_path):
    run_dir = _run_dir(core)
    cli.main(["prepare", "--run-dir", str(run_dir)])
    assert cli.main(["evaluate-test", "--run-dir", str(run_dir), "--confirm-final"]) == 2
    artifacts.atomic_json(run_dir / "frozen_bundle_manifest.json", {"bundles": {}})
    assert cli.main(["evaluate-test", "--run-dir", str(run_dir), "--confirm-final"]) == 0
    splits = artifacts.read_json(run_dir / "splits.json")
    assert set(core.calls[-1][1]) == set(splits["test_ids"])


def test_smoke_uses_only_sorted_development_patients(core, tmp_path):
    run_dir = _run_dir(core)
    cli.main(["prepare", "--run-dir", str(run_dir)])
    smoke_dir = tmp_path / "smoke"
    assert cli.main(["smoke", "--source-run", str(run_dir), "--run-dir", str(smoke_dir)]) == 0
    _, purpose, ids, splits = core.calls[-1]
    source = artifacts.read_json(run_dir / "splits.json")
    rows = {r["subject_id"]: r for r in _rows()}
    assert purpose == "smoke"
    assert len(ids) == 16 and not set(ids) & set(source["test_ids"])
    assert sum(rows[i]["label"] for i in ids) == 8
    for label in (0, 1):
        expected = sorted(
            (i for i in source["development_ids"] if rows[i]["label"] == label), key=int
        )[:8]
        assert set(expected) <= set(ids)
    saved = artifacts.read_json(smoke_dir / "config.json")
    assert saved["purpose"] == "smoke" and saved["backbones"] == ["efficientnet_b0"]
    assert saved["outer_folds"] == 2 and saved["inner_folds"] == 2
    assert saved["n_estimators"] == 10 and saved["search_iterations"] == 1
    assert saved["soft_budget_hours"] <= 1.0
    assert not (smoke_dir / "test_manifest.json").exists()
    assert splits["test_ids"] == []


def test_smoke_refuses_production_output_and_insufficient_patients(core, tmp_path, monkeypatch):
    run_dir = _run_dir(core)
    cli.main(["prepare", "--run-dir", str(run_dir)])
    assert cli.main(["smoke", "--source-run", str(run_dir), "--run-dir", str(_run_dir(core, "smoke1"))]) == 2
    assert cli.main(["smoke", "--source-run", str(run_dir), "--run-dir", str(run_dir)]) == 2
    monkeypatch.setattr(cli, "SMOKE_PER_CLASS", 1000)
    assert cli.main(["smoke", "--source-run", str(run_dir), "--run-dir", str(tmp_path / "smoke2")]) == 2
    assert not any(isinstance(c, tuple) and c[1] == "smoke" for c in core.calls)


def test_run_prepares_then_trains_without_test(core):
    run_dir = _run_dir(core)
    assert cli.main(["run", "--run-dir", str(run_dir)]) == 0
    assert core.calls[0] == "audit" and core.calls[1][0] == "train"
    assert not any(c[0] == "evaluate" for c in core.calls if isinstance(c, tuple))



def test_train_rejects_altered_fold_lists_with_old_declared_hash(core):
    run_dir = _run_dir(core)
    cli.main(["prepare", "--run-dir", str(run_dir)])
    splits = artifacts.read_json(run_dir / "splits.json")
    splits["development_ids"] = list(reversed(splits["development_ids"]))
    splits["seed"] = 1
    artifacts.atomic_json(run_dir / "splits.json", splits)
    assert cli.main(["train", "--run-dir", str(run_dir)]) == 2
    assert not any(isinstance(c, tuple) and c[0] == "train" for c in core.calls)


def test_train_rejects_splits_differing_from_registry(core):
    run_dir = _run_dir(core)
    cli.main(["prepare", "--run-dir", str(run_dir)])
    splits = artifacts.read_json(run_dir / "splits.json")
    registry = core.settings.repo_root / "runs" / "hybrid" / "protocols" / splits["cohort_hash"] / "splits.json"
    other = {**splits, "extra": 1}
    other.pop("split_hash")
    other["split_hash"] = artifacts.stable_hash(other)
    artifacts.atomic_json(registry, other)
    assert cli.main(["train", "--run-dir", str(run_dir)]) == 2


def test_train_rejects_overlapping_partitions(core):
    run_dir = _run_dir(core)
    cli.main(["prepare", "--run-dir", str(run_dir)])
    splits = artifacts.read_json(run_dir / "splits.json")
    splits["test_ids"] = splits["test_ids"] + splits["development_ids"][:1]
    splits.pop("split_hash")
    splits["split_hash"] = artifacts.stable_hash(splits)
    artifacts.atomic_json(run_dir / "splits.json", splits)
    assert cli.main(["train", "--run-dir", str(run_dir)]) == 2


def test_resumed_prepare_checks_sources_without_resplitting(core):
    run_dir = _run_dir(core)
    cli.main(["prepare", "--run-dir", str(run_dir)])
    assert core.verified == []
    assert cli.main(["prepare", "--run-dir", str(run_dir)]) == 0
    assert core.verified == [len(_rows())]
    assert core.calls.count("audit") == 1
    core.source_drift = True
    assert cli.main(["prepare", "--run-dir", str(run_dir)]) == 2


def test_smoke_audit_records_source_run(core, tmp_path):
    run_dir = _run_dir(core)
    cli.main(["prepare", "--run-dir", str(run_dir)])
    smoke_dir = tmp_path / "smoke"
    assert cli.main(["smoke", "--source-run", str(run_dir), "--run-dir", str(smoke_dir)]) == 0
    audit = artifacts.read_json(smoke_dir / "audit.json")
    assert "phenotype_sha256" not in audit
    assert audit["purpose"] == "smoke" and audit["source_run"] == str(run_dir)


def test_timeout_is_reported_as_incomplete(core, capsys):
    run_dir = _run_dir(core)
    cli.main(["prepare", "--run-dir", str(run_dir)])

    def timeout(*args, **kwargs):
        raise TimeoutError("Soft study budget reached; completed patient/fold checkpoints are preserved")

    core.training.train_development = timeout
    assert cli.main(["train", "--run-dir", str(run_dir)]) == 3
    captured = capsys.readouterr()
    assert "incomplete" in captured.err and "checkpoints" in captured.err
    assert "finished" not in captured.out
    assert cli.main(["run", "--run-dir", str(run_dir)]) == 3


def test_raw_sources_are_not_mutated_by_prepare(core):
    raw = core.settings.control_dir
    raw.mkdir(parents=True)
    scan = raw / "1_skull_stripped.nii.gz"
    scan.write_bytes(b"raw")
    before = (scan.read_bytes(), scan.stat().st_mtime_ns)
    run_dir = _run_dir(core)
    cli.main(["prepare", "--run-dir", str(run_dir)])
    assert (scan.read_bytes(), scan.stat().st_mtime_ns) == before
    assert cli.main(["prepare", "--run-dir", str(raw / "run")]) == 2
    assert [p.name for p in raw.iterdir()] == ["1_skull_stripped.nii.gz"]


def test_run_dir_cannot_be_an_allowed_root(core, tmp_path):
    import tempfile
    assert cli.main(["prepare", "--run-dir", str(core.settings.repo_root / "runs" / "hybrid")]) == 2
    assert cli.main(["prepare", "--run-dir", tempfile.gettempdir()]) == 2
