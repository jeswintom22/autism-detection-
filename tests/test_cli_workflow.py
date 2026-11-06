import dataclasses
import json
import time

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from asd_training import artifacts, cli, config as config_module, deep, evaluation, features, training
from asd_training.config import Settings
from synthetic import StubEncoder, brain_volume, write_nifti

PER_CLASS = 20
FAMILIES = ("texture_only", "saliency_only", "deep_only", "mri_hybrid", "combined")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("workflow")
    mp = pytest.MonkeyPatch()
    for name in ("ASD_DATA_ROOT", "ASD_CONTROL_DIR", "ASD_TREATMENT_DIR", "ASD_PREPROCESSED_DIR"):
        mp.delenv(name, raising=False)
    repo = root / "repo"
    (repo / "runs" / "hybrid").mkdir(parents=True)
    for label, folder in ((0, "control"), (1, "treatment")):
        for i in range(PER_CLASS):
            sid = 100 + i + label * PER_CLASS
            volume = brain_volume(sid)
            volume[volume > 0] *= 1 + 0.15 * label
            write_nifti(root / folder / f"{sid:05d}_skull_stripped.nii.gz", volume)
    weights = root / "fake_b0.pt"
    weights.write_bytes(b"tiny verified fake weight artifact")
    settings = Settings(
        repo_root=repo, control_dir=root / "control", treatment_dir=root / "treatment",
        backbones=("efficientnet_b0",), n_estimators=10,
        search_iterations=2, rf_jobs=1,
    )
    cfg = root / "config.json"
    cfg.write_text(json.dumps(settings.to_dict()))

    def stub_factory(name, settings, allow_download=True, weights_path=None):
        return StubEncoder(name, weights_path or weights, 4)

    mp.setattr(features, "FrozenEncoder", stub_factory)
    mp.setattr(deep, "FrozenEncoder", stub_factory)

    original_prepare = features.prepare_subject
    touched = []

    def recording_prepare(path, settings, hash_voxels=False):
        touched.append(str(path))
        return original_prepare(path, settings, hash_voxels)

    mp.setattr(features, "prepare_subject", recording_prepare)
    run_dir = repo / "runs" / "hybrid" / "r1"
    state = {"root": root, "settings": settings, "cfg": cfg, "run_dir": run_dir, "touched": touched,
             "weights": weights, "mp": mp}
    yield state
    mp.undo()


def rows_of(run_dir, name):
    return artifacts.read_json(run_dir / f"{name}_manifest.json")["rows"]


def test_01_prepare_creates_audited_locked_run(world):
    run_dir = world["run_dir"]
    assert cli.main(["prepare", "--config", str(world["cfg"]), "--run-dir", str(run_dir)]) == 0
    audit = artifacts.read_json(run_dir / "audit.json")
    assert audit["blocking_errors"] == [] and audit["patients"] == 2 * PER_CLASS
    assert "phenotype_sha256" not in audit
    splits = artifacts.read_json(run_dir / "splits.json")
    dev, test = rows_of(run_dir, "development"), rows_of(run_dir, "test")
    assert len(dev) == 32 and len(test) == 8
    assert {r["subject_id"] for r in dev} == set(splits["development_ids"])
    assert not {r["subject_id"] for r in dev} & {r["subject_id"] for r in test}
    registry = world["settings"].repo_root / "runs" / "hybrid" / "protocols" / splits["cohort_hash"] / "splits.json"
    assert artifacts.read_json(registry) == splits
    assert world["touched"] == [] or all("audit" not in p for p in world["touched"])
    assert cli.main(["prepare", "--config", str(world["cfg"]), "--run-dir", str(run_dir)]) == 0


def test_02_train_uses_development_only_and_freezes_five_bundles(world):
    run_dir = world["run_dir"]
    test_paths = {r["raw_path"] for r in rows_of(run_dir, "test")}
    world["mp"].setattr(cli, "_load_test_manifest", lambda *a, **k: pytest.fail("test manifest loaded in train"))
    world["touched"].clear()
    assert cli.main(["train", "--run-dir", str(run_dir)]) == 0
    assert world["touched"] and not set(world["touched"]) & test_paths
    manifest = artifacts.read_json(run_dir / "frozen_bundle_manifest.json")
    splits = artifacts.read_json(run_dir / "splits.json")
    assert set(manifest["bundles"]) == set(FAMILIES) and manifest["primary_model"] == "combined"
    assert manifest["development_ids"] == splits["development_ids"]
    for bundle in manifest["bundles"].values():
        assert set(bundle["fit_ids"]) == set(splits["development_ids"])
        assert 0.05 <= bundle["threshold"] <= 0.95
    report = artifacts.read_json(run_dir / "report_data.json")
    assert report["status"] == "frozen" and report["test"] is None
    for name in ("cv/metrics.json", "cv/outer_predictions.csv", "cv/ablation_comparison.csv", "cv/fit_scopes.json",
                 "feature_schema.json", "models/backbone_weights.pt"):
        assert (run_dir / name).is_file(), name
    assert not (run_dir / "test").exists()
    claim = world["settings"].repo_root / "runs" / "hybrid" / "protocols" / splits["cohort_hash"] / "primary_test_claim.json"
    assert not claim.exists()


def test_03_fit_scopes_never_include_validation_or_test_patients(world):
    run_dir = world["run_dir"]
    splits = artifacts.read_json(run_dir / "splits.json")
    scopes = artifacts.read_json(run_dir / "cv" / "fit_scopes.json")
    assert scopes
    folds = {f["fold"]: f for f in splits["outer_folds"]}
    for scope in scopes:
        assert not set(scope["train_ids"]) & set(splits["test_ids"])
        if "/outer-" in scope["scope"]:
            number = int(scope["scope"].split("/outer-")[1][0])
            assert not set(scope["train_ids"]) & set(folds[number]["validation_ids"])
    predictions = pd.read_csv(run_dir / "cv" / "outer_predictions.csv", dtype={"subject_id": str})
    assert not set(predictions["subject_id"]) & set(splits["test_ids"])
    primary = predictions[predictions["family"] == "combined"]
    assert sorted(primary["subject_id"]) == sorted(splits["development_ids"])
    assert (primary["prediction"] == (primary["probability"] >= primary["threshold"]).astype(int)).all()


def test_04_train_resume_returns_frozen_report_without_refitting(world):
    run_dir = world["run_dir"]
    before = artifacts.sha256_file(run_dir / "frozen_bundle_manifest.json")
    world["mp"].setattr(RandomForestClassifier, "fit", lambda *a, **k: pytest.fail("refit on resume"))
    try:
        assert cli.main(["train", "--run-dir", str(run_dir)]) == 0
    finally:
        world["mp"].undo()
        _restore(world)
    assert artifacts.sha256_file(run_dir / "frozen_bundle_manifest.json") == before


def _restore(world):
    world["mp"].setattr(features, "FrozenEncoder", lambda name, settings, allow_download=True, weights_path=None:
                        StubEncoder(name, weights_path or world["weights"], 4))
    world["mp"].setattr(deep, "FrozenEncoder", features.FrozenEncoder)
    original = features.prepare_subject
    world["mp"].setattr(features, "prepare_subject", original)


def test_05_evaluate_requires_confirmation_and_frozen_state(world):
    run_dir = world["run_dir"]
    assert cli.main(["evaluate-test", "--run-dir", str(run_dir)]) == 2
    assert not (run_dir / "test").exists()


def test_06_tampered_bundle_blocks_final_scoring(world):
    run_dir = world["run_dir"]
    target = run_dir / "models" / "combined.joblib"
    original = target.read_bytes()
    target.write_bytes(original + b"x")
    try:
        with pytest.raises(ValueError):
            evaluation.evaluate_frozen(run_dir, world["settings"], rows_of(run_dir, "test"),
                                       artifacts.read_json(run_dir / "splits.json"))
    finally:
        target.write_bytes(original)
    assert not (run_dir / "test" / "metrics.json").exists()


def test_07_final_evaluation_scores_five_frozen_bundles_with_zero_fits(world):
    run_dir = world["run_dir"]
    original_bootstrap = evaluation.bootstrap_intervals
    fit_calls = []
    local_mp = pytest.MonkeyPatch()
    try:
        local_mp.setattr(evaluation, "bootstrap_intervals", lambda y, p, threshold=0.5, n_bootstrap=2000, seed=42:
                         original_bootstrap(y, p, threshold, n_bootstrap=20, seed=seed))
        for cls in (RandomForestClassifier, ColumnTransformer, Pipeline, PCA, StandardScaler, SimpleImputer,
                    features.FeatureTransform):
            for method in ("fit", "fit_transform"):
                if hasattr(cls, method):
                    local_mp.setattr(cls, method, lambda *a, _n=f"{cls.__name__}.{method}", **k: fit_calls.append(_n))
        assert cli.main(["evaluate-test", "--run-dir", str(run_dir), "--confirm-final"]) == 0
    finally:
        local_mp.undo()
    assert fit_calls == []
    splits = artifacts.read_json(run_dir / "splits.json")
    summary = artifacts.read_json(run_dir / "test" / "metrics.json")
    manifest = artifacts.read_json(run_dir / "frozen_bundle_manifest.json")
    assert summary["n_patients"] == 8 and summary["primary_model"] == "combined"
    assert set(summary["models"]) == set(FAMILIES)
    predictions = pd.read_csv(run_dir / "test" / "predictions.csv", dtype={"subject_id": str})
    assert len(predictions) == 5 * 8
    for family in FAMILIES:
        frame = predictions[predictions["family"] == family]
        assert sorted(frame["subject_id"]) == sorted(splits["test_ids"])
        threshold = manifest["bundles"][family]["threshold"]
        assert (frame["threshold"] == threshold).all()
        assert (frame["prediction"] == (frame["probability"] >= threshold).astype(int)).all()
        metrics = summary["models"][family]
        cm = metrics["confusion_matrix"]
        assert sum(map(sum, cm)) == 8 and metrics["n_patients"] == 8
        accuracy = float((frame["prediction"] == frame["label"]).mean())
        assert metrics["accuracy"] == pytest.approx(accuracy)
        assert summary["intervals"][family]["bootstrap_samples"] == 20
    primary = summary["models"]["combined"]["accuracy"]
    assert summary["target_met"] is (primary > 0.8)
    claim = artifacts.read_json(world["settings"].repo_root / "runs" / "hybrid" / "protocols" /
                                splits["cohort_hash"] / "primary_test_claim.json")
    assert claim["status"] == "complete" and claim["freeze_hash"] == manifest["freeze_hash"]
    assert artifacts.read_json(run_dir / "report_data.json")["status"] == "complete"


def test_08_repeat_evaluation_returns_saved_result_and_report_renders(world):
    run_dir = world["run_dir"]
    before = artifacts.sha256_file(run_dir / "test" / "metrics.json")
    assert cli.main(["evaluate-test", "--run-dir", str(run_dir), "--confirm-final"]) == 0
    assert artifacts.sha256_file(run_dir / "test" / "metrics.json") == before
    assert cli.main(["report", "--run-dir", str(run_dir)]) == 0
    text = (run_dir / "report.md").read_text(encoding="utf-8")
    summary = artifacts.read_json(run_dir / "test" / "metrics.json")
    assert f"{summary['models']['combined']['accuracy']:.4f}" in text
    assert "NOT MET" in text or "MET" in text
    assert "internal, historically explored holdout" in text
    assert cli.main(["report", "--run-dir", str(run_dir)]) == 0
    assert (run_dir / "report.md").read_text(encoding="utf-8") == text


def test_09_second_run_cannot_claim_the_same_primary_test(world):
    other = world["settings"].repo_root / "runs" / "hybrid" / "r2"
    assert cli.main(["prepare", "--config", str(world["cfg"]), "--run-dir", str(other)]) == 0
    splits = artifacts.read_json(other / "splits.json")
    assert splits == artifacts.read_json(world["run_dir"] / "splits.json")
    assert cli.main(["evaluate-test", "--run-dir", str(other), "--confirm-final"]) == 2
    assert not (other / "test").exists()


def test_10_exhausted_budget_leaves_no_frozen_bundle(world):
    other = world["settings"].repo_root / "runs" / "hybrid" / "r2"
    settings = dataclasses.replace(world["settings"], soft_budget_hours=1e-9)
    splits = artifacts.read_json(other / "splits.json")
    with pytest.raises(TimeoutError):
        training.train_development(other, settings, rows_of(other, "development"), splits)
    assert not (other / "frozen_bundle_manifest.json").exists()
    assert not (other / "report_data.json").exists()


def test_11_smoke_uses_nyu_development_only_and_cannot_score_test(world, tmp_path):
    run_dir = world["run_dir"]
    smoke_dir = world["root"] / "smoke"
    assert cli.main(["smoke", "--source-run", str(run_dir), "--run-dir", str(smoke_dir)]) == 0
    source = artifacts.read_json(run_dir / "splits.json")
    settings = artifacts.read_json(smoke_dir / "config.json")
    smoke_splits = artifacts.read_json(smoke_dir / "splits.json")
    rows = rows_of(smoke_dir, "development")
    assert settings["purpose"] == "smoke" and smoke_splits["test_ids"] == []
    assert len(rows) == 16
    assert not {r["subject_id"] for r in rows} & set(source["test_ids"])
    assert {r["subject_id"] for r in rows} <= set(source["development_ids"])
    assert not (smoke_dir / "test_manifest.json").exists()
    manifest = artifacts.read_json(smoke_dir / "frozen_bundle_manifest.json")
    assert manifest["purpose"] == "smoke"
    assert "phenotype_sha256" not in artifacts.read_json(smoke_dir / "audit.json")
    assert cli.main(["evaluate-test", "--run-dir", str(smoke_dir), "--confirm-final"]) == 2
    assert not (smoke_dir / "test").exists()
    assert cli.main(["smoke", "--source-run", str(run_dir), "--run-dir", str(run_dir / "inside")]) == 2


def test_12_changed_raw_data_blocks_resumed_prepare(world):
    scan = next((world["root"] / "control").glob("*.nii.gz"))
    original = scan.read_bytes()
    scan.write_bytes(original + b"\0")
    try:
        assert cli.main(["prepare", "--config", str(world["cfg"]), "--run-dir", str(world["run_dir"])]) == 2
    finally:
        scan.write_bytes(original)
    assert cli.main(["prepare", "--config", str(world["cfg"]), "--run-dir", str(world["run_dir"])]) == 0
