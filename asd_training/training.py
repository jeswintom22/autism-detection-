import json
import os
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler
from threadpoolctl import threadpool_limits

from .artifacts import atomic_csv, atomic_json, ensure_safe_output, read_json, sha256_file, stable_hash
from .config import config_hash
from .dataset import verify_sources
from .evaluation import best_threshold, patient_metrics
from .features import (FAMILIES, FeatureTransform, extract_deep_cached, extract_features,
                       extract_static_parallel, load_patient_cnn)
from .finetune import FineTunedEncoder
from .splits import validate_development_splits


LIMITATIONS = [
    "The primary combined model fuses MRI deep, texture, and saliency features; no clinical variables are used.",
    "The locked patient holdout is internal and historically explored in earlier experiments, not external validation.",
    "Scanner sites overlap development/test. Generalization to unseen sites is not established.",
    "RAS orientation and 1 mm resampling do not constitute anatomical atlas registration.",
    "Exact identity/content guards do not prove absence of near-duplicates or family relationships.",
    "This is a research classifier, not a clinically validated diagnostic tool.",
    "Greater than 80% patient accuracy is an experimental target, not a guaranteed result.",
    "Development scores include architecture/threshold selection; only the designated frozen test score is final.",
]


def atomic_joblib(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def _check_budget(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Soft study budget reached; completed patient/fold checkpoints are preserved")


def _forest(settings, params, seed):
    return RandomForestClassifier(
        n_estimators=settings.n_estimators, bootstrap=True, class_weight="balanced_subsample",
        random_state=seed, n_jobs=settings.rf_jobs, **params,
    )


def _probability(model, matrix):
    classes = model.classes_.tolist()
    if classes != [0, 1]:
        raise ValueError("Random Forest must have control=0 and ASD=1 classes")
    jobs = model.n_jobs
    try:
        model.n_jobs = 1
        return model.predict_proba(matrix)[:, classes.index(1)]
    finally:
        model.n_jobs = jobs


def _scoped_transform(frame, ids, forbidden, settings, backbone, family, scope, scopes):
    if len(set(ids)) != len(ids) or set(ids) & set(forbidden) or not set(ids) <= set(frame.index):
        raise ValueError(f"Invalid/leaky patient fit membership: {scope}")
    transformer = FeatureTransform(settings, backbone, family)
    values = transformer.fit_transform(frame.loc[ids])
    if transformer.fit_ids_ != list(ids):
        raise ValueError("Feature-transform fit patient provenance mismatch")
    scopes.append({"kind": "feature_transform", "scope": scope, "family": family,
                   "backbone": backbone, "train_ids": list(ids), "forbidden_ids": sorted(forbidden)})
    return transformer, values


def _fit_rf(matrix, labels, ids, settings, params, seed, scope, scopes):
    scopes.append({"kind": "random_forest", "scope": scope, "train_ids": list(ids),
                   "parameters": params, "seed": seed})
    return _forest(settings, params, seed).fit(matrix, labels.loc[ids].to_numpy(int))


def _inner_search(frame, labels, folds, settings, backbone, family, forbidden, scope, scopes,
                  deadline, fixed_params=None):
    candidates = [fixed_params] if fixed_params is not None else list(ParameterSampler({
        "max_depth": [None, 10, 20], "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.25, 0.5], "max_samples": [None, 0.8],
    }, n_iter=settings.search_iterations, random_state=42))
    prepared = []
    for fold in folds:
        _check_budget(deadline)
        train, valid = fold["train_ids"], fold["validation_ids"]
        transform, matrix = _scoped_transform(
            frame, train, set(forbidden) | set(valid), settings, backbone, family,
            f"{scope}/inner-{fold['fold']}", scopes,
        )
        prepared.append((fold, matrix, transform.transform(frame.loc[valid])))
    results, predictions = [], []
    for candidate_index, params in enumerate(candidates):
        _check_budget(deadline)
        oof, scores, aucs = {}, [], []
        for fold, matrix, validation_matrix in prepared:
            train, valid = fold["train_ids"], fold["validation_ids"]
            model = _fit_rf(
                matrix, labels, train, settings, params, settings.seed + fold["fold"],
                f"{scope}/candidate-{candidate_index}/inner-{fold['fold']}", scopes,
            )
            p = _probability(model, validation_matrix)
            measured = patient_metrics(labels.loc[valid].to_numpy(int), p, 0.5)
            scores.append(measured["balanced_accuracy"])
            aucs.append(measured["roc_auc"])
            oof.update(zip(valid, p.tolist()))
        results.append({
            "scope": scope, "backbone": backbone, "family": family, "candidate": candidate_index,
            "parameters": params, "mean_balanced_accuracy": float(np.mean(scores)),
            "mean_roc_auc": float(np.mean(aucs)),
        })
        predictions.append(oof)
    winner = min(range(len(results)), key=lambda i: (
        -results[i]["mean_balanced_accuracy"], -results[i]["mean_roc_auc"],
        json.dumps(results[i]["parameters"], sort_keys=True),
    ))
    oof = predictions[winner]
    expected = {sid for fold in folds for sid in fold["validation_ids"]}
    if set(oof) != expected:
        raise ValueError("Incomplete inner patient OOF predictions")
    ids = sorted(oof, key=int)
    threshold = best_threshold(labels.loc[ids].to_numpy(int), [oof[sid] for sid in ids])
    return results[winner]["parameters"], threshold, oof, results


def _outer_result(frame, labels, mapping, fold, settings, backbone, family, test_ids,
                  deadline, fixed_params=None):
    scopes = []
    train, valid = fold["train_ids"], fold["validation_ids"]
    params, threshold, _, search = _inner_search(
        frame, labels, fold["inner_folds"], settings, backbone, family,
        set(test_ids) | set(valid), f"{backbone}/{family}/outer-{fold['fold']}", scopes,
        deadline, fixed_params,
    )
    transform, matrix = _scoped_transform(
        frame, train, set(test_ids) | set(valid), settings, backbone, family,
        f"{backbone}/{family}/outer-{fold['fold']}/refit", scopes,
    )
    forest = _fit_rf(matrix, labels, train, settings, params, settings.seed + fold["fold"],
                     f"{backbone}/{family}/outer-{fold['fold']}/refit", scopes)
    p = _probability(forest, transform.transform(frame.loc[valid]))
    predicted = p >= threshold
    predictions = [{
        "subject_id": sid, "backbone": backbone, "family": family, "fold": fold["fold"],
        "label": int(labels[sid]), "probability": float(p[i]), "threshold": threshold,
        "prediction": int(predicted[i]),
    } for i, sid in enumerate(valid)]
    return {
        "parameters": params, "threshold": threshold,
        "metrics": patient_metrics(labels.loc[valid].to_numpy(int), p, threshold),
        "predictions": predictions, "scopes": scopes, "search": search,
        "outer_bundle": {"transform": transform, "forest": forest, "threshold": threshold},
    }


def _summary(results, backbone, family):
    summary = {"backbone": backbone, "family": family, "folds": len(results)}
    for key in ("accuracy", "balanced_accuracy", "roc_auc", "average_precision", "sensitivity", "specificity", "f1"):
        values = [item["metrics"][key] for item in results]
        summary[f"mean_{key}"] = float(np.mean(values))
        summary[f"std_{key}"] = float(np.std(values))
    return summary


def _get_fold_encoder(run_dir, settings, fold, rows_by_id, labels, cache_dir, deadline):
    backbone = (settings.finetune or {}).get("backbone", "efficientnet_b5")
    checkpoint = run_dir / "cv" / "encoders" / f"outer-{fold['fold']}.pt"
    meta_path = checkpoint.with_suffix(".json")
    expected_labels = stable_hash({sid: int(labels[sid]) for sid in fold["train_ids"]})
    if checkpoint.is_file() and meta_path.is_file():
        meta = read_json(meta_path)
        if (meta.get("labels_hash") != expected_labels
                or meta.get("metadata", {}).get("backbone") != backbone
                or set(meta.get("train_ids", [])) | set(meta.get("val_ids", [])) != set(fold["train_ids"])
                or set(meta.get("train_ids", [])) & set(meta.get("val_ids", []))):
            raise ValueError(f"Fold encoder checkpoint provenance mismatch: outer-{fold['fold']}")
        if sha256_file(checkpoint) != meta.get("sha256"):
            raise ValueError("Fold encoder checkpoint changed")
        print(f"finetune: reusing outer-{fold['fold']} encoder", flush=True)
        return FineTunedEncoder.load(checkpoint, settings)
    encoder = FineTunedEncoder(settings, backbone)
    patients = [
        (sid, load_patient_cnn(rows_by_id[sid], settings, cache_dir))
        for sid in fold["train_ids"]
    ]
    fold_labels = {sid: int(labels[sid]) for sid in fold["train_ids"]}
    encoder.finetune(patients, fold_labels, settings.finetune, checkpoint, deadline)
    encoder.close()
    del patients
    return FineTunedEncoder.load(checkpoint, settings)


def _train_finetuned(run_dir, settings, development_rows, splits, deadline):
    started = time.monotonic()
    cache_dir = run_dir / "cache" / "subject_features"
    backbone = (settings.finetune or {}).get("backbone", "efficientnet_b5")
    frame_static, schema = extract_static_parallel(
        development_rows, settings, cache_dir,
        workers=settings.extract_workers, deadline=deadline,
    )
    atomic_json(run_dir / "feature_schema.json", schema)
    mapping = {row["subject_id"]: row for row in development_rows}
    labels = pd.Series({sid: row["label"] for sid, row in mapping.items()}, dtype=int)
    rows_by_id = {sid: row for sid, row in mapping.items()}
    study_hash = stable_hash({
        "config": config_hash(settings), "split": splits["split_hash"],
        "patients": development_rows, "features": schema, "finetune": settings.finetune,
    })
    all_results, all_scopes, all_search = {}, [], []
    fold_frames, encoders = {}, {}

    def encoder_for(fold):
        if fold["fold"] not in encoders:
            encoders[fold["fold"]] = _get_fold_encoder(
                run_dir, settings, fold, rows_by_id, labels, cache_dir, deadline)
        return encoders[fold["fold"]]

    def fold_frame(fold):
        if fold["fold"] not in fold_frames:
            encoder = encoder_for(fold)
            ids = list(fold["train_ids"]) + list(fold["validation_ids"])
            rows = [rows_by_id[sid] for sid in ids]
            deep = extract_deep_cached(rows, settings, cache_dir, encoder)
            frame = frame_static.loc[ids].join(deep)
            fold_frames[fold["fold"]] = frame
        return fold_frames[fold["fold"]]

    def checkpoint(family, fold, fixed=None):
        _check_budget(deadline)
        path = run_dir / "cv" / "checkpoints" / f"{backbone}_{family}_fold{fold['fold']}.joblib"
        if path.is_file():
            saved = joblib.load(path)
            if saved["study_hash"] != study_hash:
                raise ValueError("Fold checkpoint belongs to a different study")
            result = saved["result"]
        else:
            print(f"RF: {backbone} {family} outer fold {fold['fold']}/{settings.outer_folds}", flush=True)
            result = _outer_result(
                fold_frame(fold), labels, mapping, fold, settings, backbone, family,
                splits["test_ids"], deadline, fixed,
            )
            atomic_joblib(path, {"study_hash": study_hash, "result": result})
        all_scopes.extend(result["scopes"])
        all_search.extend(result["search"])
        return {key: value for key, value in result.items() if key != "outer_bundle"}

    try:
        results = [checkpoint("combined", fold) for fold in splits["outer_folds"]]
        all_results[(backbone, "combined")] = results
        comparisons = [_summary(results, backbone, "combined")]
        atomic_json(run_dir / "cv" / "progress.json", {
            "status": "development_in_progress", "completed_backbone_comparisons": comparisons,
            "test_evaluated": False,
        })
        for family in FAMILIES:
            if family != "combined":
                all_results[(backbone, family)] = [
                    checkpoint(family, fold, results[i]["parameters"])
                    for i, fold in enumerate(splits["outer_folds"])
                ]
        ablations = [_summary(all_results[(backbone, family)], backbone, family) for family in FAMILIES]
        predictions = [row for items in all_results.values() for item in items for row in item["predictions"]]
        atomic_csv(run_dir / "cv" / "outer_predictions.csv", predictions)
        atomic_csv(run_dir / "cv" / "ablation_comparison.csv", ablations)
        atomic_json(run_dir / "cv" / "metrics.json", {
            "backbone_comparison": comparisons, "ablation_comparison": ablations,
            "selected_backbone": backbone,
        })
        final_checkpoint = run_dir / "models" / "finetuned_encoder.pt"
        if final_checkpoint.is_file():
            meta = read_json(final_checkpoint.with_suffix(".json"))
        else:
            final_encoder = FineTunedEncoder(settings, backbone)
            patients = [
                (sid, load_patient_cnn(rows_by_id[sid], settings, cache_dir))
                for sid in splits["development_ids"]
            ]
            meta = final_encoder.finetune(
                patients, {sid: int(labels[sid]) for sid in splits["development_ids"]},
                settings.finetune, final_checkpoint, deadline,
            )
            final_encoder.close()
            del patients
        final_encoder = FineTunedEncoder.load(final_checkpoint, settings)
        try:
            deep_all = extract_deep_cached(development_rows, settings, cache_dir, final_encoder)
        finally:
            final_encoder.close()
        frame_all = frame_static.join(deep_all)
        params, _, _, final_search = _inner_search(
            frame_all, labels, splits["final_inner_folds"], settings, backbone, "combined",
            splits["test_ids"], "final-development-search", all_scopes, deadline,
        )
        all_search.extend(final_search)
        bundles = {}
        for family in FAMILIES:
            _check_budget(deadline)
            oof = [row for item in all_results[(backbone, family)] for row in item["predictions"]]
            if len(oof) != len(mapping) or {row["subject_id"] for row in oof} != set(mapping):
                raise ValueError("Final threshold needs one OOF probability per development patient")
            threshold = best_threshold([row["label"] for row in oof], [row["probability"] for row in oof])
            ids = splits["development_ids"]
            transform, matrix = _scoped_transform(
                frame_all, ids, splits["test_ids"], settings, backbone, family,
                f"final/{family}", all_scopes,
            )
            forest = _fit_rf(matrix, labels, ids, settings, params, settings.seed,
                             f"final/{family}", all_scopes)
            path = run_dir / "models" / f"{family}.joblib"
            bundle = {"transform": transform, "forest": forest, "threshold": threshold,
                      "backbone": backbone, "family": family, "fit_ids": list(ids),
                      "schema_hash": schema["schema_hash"], "config_hash": config_hash(settings)}
            atomic_joblib(path, bundle)
            bundles[family] = {"file": str(path.relative_to(run_dir)), "sha256": sha256_file(path),
                               "threshold": threshold, "fit_ids": list(ids)}
        weight_sha = sha256_file(final_checkpoint)
        manifest = {
            "schema_version": 1, "purpose": settings.purpose, "primary_model": "combined",
            "config_hash": config_hash(settings), "split_hash": splits["split_hash"],
            "cohort_hash": splits["cohort_hash"], "development_ids": splits["development_ids"],
            "backbone": backbone,
            "encoder": {
                **meta["metadata"], "checkpoint": "models/finetuned_encoder.pt",
                "sha256": weight_sha, "weight_sha256": weight_sha,
                "data_hash": meta["data_hash"], "train_ids": meta["train_ids"],
                "val_ids": meta["val_ids"],
            },
            "feature_schema": schema,
            "bundles": bundles,
            "weights": {"file": "models/finetuned_encoder.pt", "sha256": weight_sha},
        }
        manifest["freeze_hash"] = stable_hash(manifest)
        atomic_json(run_dir / "cv" / "fit_scopes.json", all_scopes)
        atomic_json(run_dir / "cv" / "inner_search.json", all_search)
        atomic_csv(run_dir / "cv" / "inner_search.csv", [
            {**row, "parameters": json.dumps(row["parameters"], sort_keys=True)} for row in all_search
        ])
        report = {
            "status": "frozen", "primary_model": "combined",
            "development": {"backbone_comparison": comparisons, "ablation_comparison": ablations},
            "test": None, "limitations": LIMITATIONS, "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(run_dir / "report_data.json", report)
        atomic_json(run_dir / "frozen_bundle_manifest.json", manifest)
        print(f"Frozen fine-tuned {backbone} combined model and four ablations; "
              "test has not been evaluated.", flush=True)
        return report
    finally:
        for encoder in encoders.values():
            encoder.close()


def train_development(run_dir, settings, development_rows, splits):
    run_dir = ensure_safe_output(Path(run_dir), settings)
    validate_development_splits(development_rows, splits)
    if (splits.get("purpose") != settings.purpose or len(splits["outer_folds"]) != settings.outer_folds
            or len(splits["final_inner_folds"]) != settings.inner_folds
            or any(len(fold["inner_folds"]) != settings.inner_folds for fold in splits["outer_folds"])):
        raise ValueError("Patient fold counts/purpose differ from the frozen configuration")
    if {row["subject_id"] for row in development_rows} != set(splits["development_ids"]):
        raise ValueError("Training input must contain exactly the frozen development patients")
    if set(splits["development_ids"]) & set(splits["test_ids"]):
        raise ValueError("Patient holdout leakage")
    verify_sources(development_rows)
    if (run_dir / "frozen_bundle_manifest.json").is_file():
        frozen = read_json(run_dir / "frozen_bundle_manifest.json")
        if frozen["config_hash"] != config_hash(settings) or frozen["split_hash"] != splits["split_hash"]:
            raise ValueError("Incompatible frozen-run resume")
        for bundle in frozen["bundles"].values():
            if sha256_file(run_dir / bundle["file"]) != bundle["sha256"]:
                raise ValueError("Frozen model changed")
        return read_json(run_dir / "report_data.json")
    if (settings.finetune or {}).get("enabled"):
        deadline = time.monotonic() + settings.soft_budget_hours * 3600
        with threadpool_limits(limits=2):
            return _train_finetuned(run_dir, settings, development_rows, splits, deadline)
    started = time.monotonic()
    deadline = started + settings.soft_budget_hours * 3600
    with threadpool_limits(limits=2):
        frame, encoders, schema = extract_features(
            development_rows, settings, run_dir / "cache" / "subject_features",
            deadline=deadline,
        )
        atomic_json(run_dir / "feature_schema.json", schema)
        mapping = {row["subject_id"]: row for row in development_rows}
        labels = pd.Series({sid: row["label"] for sid, row in mapping.items()}, dtype=int)
        study_hash = stable_hash({
            "config": config_hash(settings), "split": splits["split_hash"],
            "patients": development_rows, "features": schema,
        })
        all_results, all_scopes, all_search = {}, [], []

        def checkpoint(backbone, family, fold, fixed=None):
            _check_budget(deadline)
            path = run_dir / "cv" / "checkpoints" / f"{backbone}_{family}_fold{fold['fold']}.joblib"
            if path.is_file():
                saved = joblib.load(path)
                if saved["study_hash"] != study_hash:
                    raise ValueError("Fold checkpoint belongs to a different study")
                result = saved["result"]
            else:
                print(f"RF: {backbone} {family} outer fold {fold['fold']}/{settings.outer_folds}", flush=True)
                result = _outer_result(
                    frame, labels, mapping, fold, settings, backbone, family,
                    splits["test_ids"], deadline, fixed,
                )
                atomic_joblib(path, {"study_hash": study_hash, "result": result})
            all_scopes.extend(result["scopes"])
            all_search.extend(result["search"])
            return {key: value for key, value in result.items() if key != "outer_bundle"}

        comparisons = []
        for backbone in settings.backbones:
            results = [checkpoint(backbone, "combined", fold) for fold in splits["outer_folds"]]
            all_results[(backbone, "combined")] = results
            comparisons.append(_summary(results, backbone, "combined"))
            atomic_json(run_dir / "cv" / "progress.json", {
                "status": "development_in_progress", "completed_backbone_comparisons": comparisons,
                "test_evaluated": False,
            })
        selected = min(comparisons, key=lambda row: (
            -row["mean_balanced_accuracy"], -row["mean_roc_auc"],
            row["backbone"] != "efficientnet_b0",
        ))["backbone"]
        for family in FAMILIES:
            if family != "combined":
                all_results[(selected, family)] = [
                    checkpoint(selected, family, fold, all_results[(selected, "combined")][i]["parameters"])
                    for i, fold in enumerate(splits["outer_folds"])
                ]
        ablations = [_summary(all_results[(selected, family)], selected, family) for family in FAMILIES]
        predictions = [row for results in all_results.values() for item in results for row in item["predictions"]]
        atomic_csv(run_dir / "cv" / "outer_predictions.csv", predictions)
        atomic_csv(run_dir / "cv" / "ablation_comparison.csv", ablations)
        atomic_json(run_dir / "cv" / "metrics.json", {
            "backbone_comparison": comparisons, "ablation_comparison": ablations,
            "selected_backbone": selected,
        })
        params, _, _, final_search = _inner_search(
            frame, labels, splits["final_inner_folds"], settings, selected, "combined",
            splits["test_ids"], "final-development-search", all_scopes, deadline,
        )
        all_search.extend(final_search)
        bundles = {}
        for family in FAMILIES:
            _check_budget(deadline)
            oof = [row for item in all_results[(selected, family)] for row in item["predictions"]]
            if len(oof) != len(mapping) or {row["subject_id"] for row in oof} != set(mapping):
                raise ValueError("Final threshold needs one OOF probability per development patient")
            threshold = best_threshold([row["label"] for row in oof], [row["probability"] for row in oof])
            ids = splits["development_ids"]
            transform, matrix = _scoped_transform(
                frame, ids, splits["test_ids"], settings, selected, family, f"final/{family}", all_scopes,
            )
            forest = _fit_rf(matrix, labels, ids, settings, params, settings.seed, f"final/{family}", all_scopes)
            path = run_dir / "models" / f"{family}.joblib"
            bundle = {"transform": transform, "forest": forest, "threshold": threshold,
                      "backbone": selected, "family": family, "fit_ids": list(ids),
                      "schema_hash": schema["schema_hash"], "config_hash": config_hash(settings)}
            atomic_joblib(path, bundle)
            bundles[family] = {"file": str(path.relative_to(run_dir)), "sha256": sha256_file(path),
                               "threshold": threshold, "fit_ids": list(ids)}
        weights = run_dir / "models" / "backbone_weights.pt"
        shutil.copyfile(encoders[selected]["weights_path"], weights)
        manifest = {
            "schema_version": 1, "purpose": settings.purpose, "primary_model": "combined",
            "config_hash": config_hash(settings), "split_hash": splits["split_hash"],
            "cohort_hash": splits["cohort_hash"], "development_ids": splits["development_ids"],
            "backbone": selected, "encoder": encoders[selected], "feature_schema": schema,
            "bundles": bundles,
            "weights": {"file": "models/backbone_weights.pt", "sha256": sha256_file(weights)},
        }
        manifest["freeze_hash"] = stable_hash(manifest)
        atomic_json(run_dir / "cv" / "fit_scopes.json", all_scopes)
        atomic_json(run_dir / "cv" / "inner_search.json", all_search)
        atomic_csv(run_dir / "cv" / "inner_search.csv", [
            {**row, "parameters": json.dumps(row["parameters"], sort_keys=True)} for row in all_search
        ])
        report = {
            "status": "frozen", "primary_model": "combined",
            "development": {"backbone_comparison": comparisons, "ablation_comparison": ablations},
            "test": None, "limitations": LIMITATIONS, "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(run_dir / "report_data.json", report)
        atomic_json(run_dir / "frozen_bundle_manifest.json", manifest)
        print(f"Frozen {selected} combined model and four ablations; test has not been evaluated.", flush=True)
        return report
