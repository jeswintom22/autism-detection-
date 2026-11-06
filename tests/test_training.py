import dataclasses
import hashlib
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from asd_training import features, splits as splits_module, training
from asd_training.config import Settings


@pytest.fixture
def settings():
    return Settings(purpose="smoke", outer_folds=2, inner_folds=2, n_estimators=10, search_iterations=2,
                    backbones=("efficientnet_b0",), rf_jobs=1)


def fp(prefix, i):
    return hashlib.sha256(f"{prefix}{i}".encode()).hexdigest()


def make_rows(n_per_class=20):
    rows = []
    for i in range(2 * n_per_class):
        rows.append({"subject_id": str(100 + i), "label": int(i >= n_per_class), "site_id": "S",
                     "source_sha256": fp("s", i), "voxel_sha256": fp("v", i)})
    return rows


def make_frame(rows, seed=0, deep_dim=12, texture_dim=36, saliency_dim=12):
    rng = np.random.default_rng(seed)
    n = len(rows)
    labels = np.array([r["label"] for r in rows])
    data = {}
    for i in range(deep_dim):
        data[f"deep__efficientnet_b0__sagittal__mean__{i:04d}"] = rng.normal(size=n) + (0.8 * labels if i == 0 else 0)
    for i in range(texture_dim):
        data[f"texture__sagittal__mean__{i}"] = rng.normal(size=n) + (0.5 * labels if i == 0 else 0)
    for i in range(saliency_dim):
        data[f"saliency__sagittal__mean__{i}"] = rng.normal(size=n) + (0.3 * labels if i == 0 else 0)
    frame = pd.DataFrame(data, index=pd.Index([r["subject_id"] for r in rows], name="subject_id"))
    return frame, pd.Series(labels, index=frame.index)


@pytest.fixture
def study(settings):
    rows = make_rows()
    frame, labels = make_frame(rows)
    split = splits_module.create_development_folds(rows, settings)
    mapping = {r["subject_id"]: r for r in rows}
    return rows, frame, labels, split, mapping


def far_deadline():
    return time.monotonic() + 600


@pytest.mark.parametrize("family,width", [
    ("saliency_only", 12), ("texture_only", 36), ("deep_only", 12), ("mri_hybrid", 48), ("combined", 60),
])
def test_family_output_widths(settings, study, family, width):
    _, frame, _, _, _ = study
    transform = features.FeatureTransform(settings, "efficientnet_b0", family)
    matrix = transform.fit_transform(frame.iloc[:20])
    assert matrix.shape == (20, width) and matrix.dtype == np.float32
    assert isinstance(transform.transformer_, ColumnTransformer)
    assert transform.fit_ids_ == list(frame.index[:20])
    assert transform.transform(frame.iloc[20:]).shape == (20, width)


def test_pca_is_clamped_by_training_size(settings, study):
    _, frame, _, _, _ = study
    transform = features.FeatureTransform(settings, "efficientnet_b0", "deep_only")
    assert transform.fit_transform(frame.iloc[:6]).shape[1] == 5
    big = dataclasses.replace(settings, pca_components=3)
    assert features.FeatureTransform(big, "efficientnet_b0", "deep_only").fit_transform(frame).shape[1] == 3


def test_unknown_family_and_unfitted_transform_are_rejected(settings, study):
    with pytest.raises(ValueError):
        features.FeatureTransform(settings, "efficientnet_b0", "everything")
    with pytest.raises(ValueError):
        features.FeatureTransform(settings, "efficientnet_b0", "combined").transform(study[1])


def test_saliency_only_is_passthrough(settings):
    train = pd.DataFrame({f"saliency__sagittal__mean__{i}": [float(i), 2.0 * i, 3.0 * i]
                          for i in range(12)}, index=["1", "2", "3"])
    transform = features.FeatureTransform(settings, "efficientnet_b0", "saliency_only").fit(train)
    out = transform.transform(train)
    assert out.shape == (3, 12)
    assert np.allclose(out, train.to_numpy())


def test_prohibited_and_nonfinite_columns_never_enter(settings, study):
    _, frame, _, _, _ = study
    for name in ("label", "site_id", "subject_id", "DX_GROUP", "ADOS_TOTAL", "diagnosis", "age", "fiq"):
        poisoned = frame.assign(**{name: 1.0})
        with pytest.raises(ValueError, match="allowlisted"):
            features.FeatureTransform(settings, "efficientnet_b0", "combined").fit(poisoned)
    broken = frame.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        features.FeatureTransform(settings, "efficientnet_b0", "combined").fit(broken)
    with pytest.raises(ValueError):
        features.FeatureTransform(settings, "efficientnet_b0", "combined").fit(
            frame[[c for c in frame if not c.startswith("saliency__")]])
    with pytest.raises(ValueError):
        features.FeatureTransform(settings, "efficientnet_b0", "combined").fit(
            pd.concat([frame.iloc[:3], frame.iloc[:3]]))


def test_matrix_columns_exclude_target_by_construction(settings, study):
    rows, frame, labels, _, _ = study
    matrix = features.FeatureTransform(settings, "efficientnet_b0", "combined").fit_transform(frame)
    assert matrix.shape[1] == 60
    assert not any(np.array_equal(matrix[:, i], labels.to_numpy(float)) for i in range(matrix.shape[1]))


def test_extreme_validation_values_cannot_change_fitted_transform(settings, study):
    rows, frame, labels, split, _ = study
    fold = split["outer_folds"][0]
    train, valid = fold["train_ids"], fold["validation_ids"]
    spoiled = frame.copy()
    spoiled.loc[valid] = spoiled.loc[valid] * 1e6
    spoiled.loc[valid, "saliency__sagittal__mean__0"] = 1e9
    scopes_a, scopes_b = [], []
    ta, ma = training._scoped_transform(frame, train, valid, settings, "efficientnet_b0", "combined", "a", scopes_a)
    tb, mb = training._scoped_transform(spoiled, train, valid, settings, "efficientnet_b0", "combined", "b", scopes_b)
    assert np.array_equal(ma, mb)
    pa, pb = ta.transformer_.named_transformers_, tb.transformer_.named_transformers_
    assert np.array_equal(pa["deep"].named_steps["pca"].components_, pb["deep"].named_steps["pca"].components_)
    assert np.array_equal(pa["deep"].named_steps["scale"].mean_, pb["deep"].named_steps["scale"].mean_)
    assert ta.fit_ids_ == list(train) and not set(ta.fit_ids_) & set(valid)


def test_scoped_transform_refuses_leaky_or_invalid_membership(settings, study):
    _, frame, _, _, _ = study
    ids = list(frame.index[:10])
    for bad_ids, forbidden in ((ids, {ids[0]}), (ids + [ids[0]], set()), (ids + ["nope"], set())):
        with pytest.raises(ValueError):
            training._scoped_transform(frame, bad_ids, forbidden, settings, "efficientnet_b0", "combined", "x", [])
    scopes = []
    training._scoped_transform(frame, ids, set(frame.index[10:]), settings, "efficientnet_b0", "combined", "ok", scopes)
    assert scopes[0]["train_ids"] == ids and scopes[0]["forbidden_ids"] == sorted(frame.index[10:])


def test_inner_search_fit_scopes_and_threshold_receive_only_inner_patients(settings, study, monkeypatch):
    rows, frame, labels, split, mapping = study
    outer = split["outer_folds"][0]
    seen_fits, seen_thresholds = [], []
    original_fit = features.FeatureTransform.fit

    def spy_fit(self, frame_, y=None):
        seen_fits.append(list(frame_.index))
        return original_fit(self, frame_, y)

    original_threshold = training.best_threshold

    def spy_threshold(y, p):
        seen_thresholds.append(list(y))
        return original_threshold(y, p)

    monkeypatch.setattr(features.FeatureTransform, "fit", spy_fit)
    monkeypatch.setattr(training, "best_threshold", spy_threshold)
    scopes = []
    forbidden = set(split["test_ids"]) | set(outer["validation_ids"])
    params, threshold, oof, results = training._inner_search(
        frame, labels, outer["inner_folds"], settings, "efficientnet_b0", "combined", forbidden,
        "scope", scopes, far_deadline())
    inner_train = [f["train_ids"] for f in outer["inner_folds"]]
    assert seen_fits == inner_train
    assert not set().union(*map(set, seen_fits)) & set(outer["validation_ids"])
    assert set(oof) == set(outer["train_ids"])
    assert len(results) == settings.search_iterations
    transform_scopes = [s for s in scopes if s["kind"] == "feature_transform"]
    forest_scopes = [s for s in scopes if s["kind"] == "random_forest"]
    assert [s["train_ids"] for s in transform_scopes] == inner_train
    assert len(forest_scopes) == settings.search_iterations * len(inner_train)
    for scope in forest_scopes:
        assert scope["train_ids"] in inner_train
    assert len(seen_thresholds) == 1 and len(seen_thresholds[0]) == len(outer["train_ids"])
    assert seen_thresholds[0] == [int(labels[i]) for i in sorted(oof, key=int)]
    assert 0.05 <= threshold <= 0.95


def test_inner_search_tie_break_prefers_canonical_parameters(settings, study):
    _, frame, labels, split, _ = study
    outer = split["outer_folds"][0]
    constant = pd.Series(0, index=labels.index)
    constant.iloc[::2] = 1
    flat_frame = frame * 0 + 1
    scopes = []
    params, _, _, results = training._inner_search(
        flat_frame, constant, outer["inner_folds"], settings, "efficientnet_b0", "saliency_only",
        set(), "tie", scopes, far_deadline())
    best = min(results, key=lambda r: (-r["mean_balanced_accuracy"], -r["mean_roc_auc"],
                                       __import__("json").dumps(r["parameters"], sort_keys=True)))
    assert params == best["parameters"]


def test_outer_result_predicts_only_untouched_validation_patients(settings, study):
    rows, frame, labels, split, mapping = study
    outer = split["outer_folds"][1]
    result = training._outer_result(frame, labels, mapping, outer, settings, "efficientnet_b0", "combined",
                                    split["test_ids"], far_deadline())
    predicted_ids = [p["subject_id"] for p in result["predictions"]]
    assert sorted(predicted_ids) == sorted(outer["validation_ids"])
    for scope in result["scopes"]:
        assert not set(scope["train_ids"]) & set(outer["validation_ids"])
        assert not set(scope["train_ids"]) & set(split["test_ids"])
    refit = [s for s in result["scopes"] if s["scope"].endswith("/refit")]
    assert {s["kind"] for s in refit} == {"feature_transform", "random_forest"}
    assert all(s["train_ids"] == outer["train_ids"] for s in refit)
    assert result["outer_bundle"]["transform"].fit_ids_ == outer["train_ids"]
    threshold = result["threshold"]
    for row in result["predictions"]:
        assert row["prediction"] == int(row["probability"] >= threshold)
        assert row["threshold"] == threshold and row["label"] == int(labels[row["subject_id"]])
    assert result["metrics"]["n_patients"] == len(outer["validation_ids"])
    cm = result["metrics"]["confusion_matrix"]
    assert sum(map(sum, cm)) == len(outer["validation_ids"])


def test_fixed_parameters_are_reused_for_ablations(settings, study):
    rows, frame, labels, split, mapping = study
    outer = split["outer_folds"][0]
    chosen = {"max_depth": 10, "min_samples_leaf": 2, "max_features": "sqrt", "max_samples": None}
    result = training._outer_result(frame, labels, mapping, outer, settings, "efficientnet_b0", "saliency_only",
                                    split["test_ids"], far_deadline(), chosen)
    assert result["parameters"] == chosen and len(result["search"]) == 1


def test_expired_budget_raises_before_any_fit(settings, study):
    _, frame, labels, split, _ = study
    outer = split["outer_folds"][0]
    with pytest.raises(TimeoutError):
        training._inner_search(frame, labels, outer["inner_folds"], settings, "efficientnet_b0", "combined",
                               set(), "late", [], time.monotonic() - 1)


def test_summary_requires_all_fold_results(settings, study):
    rows, frame, labels, split, mapping = study
    results = [training._outer_result(frame, labels, mapping, fold, settings, "efficientnet_b0", "combined",
                                      split["test_ids"], far_deadline()) for fold in split["outer_folds"]]
    summary = training._summary(results, "efficientnet_b0", "combined")
    assert summary["folds"] == len(split["outer_folds"]) == 2
    assert summary["mean_balanced_accuracy"] == pytest.approx(
        np.mean([r["metrics"]["balanced_accuracy"] for r in results]))
