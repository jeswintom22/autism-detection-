import math

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def _inputs(y, probabilities):
    labels = np.asarray(y)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape:
        raise ValueError("Expected equal-length patient label/probability vectors")
    if not set(labels.tolist()) <= {0, 1}:
        raise ValueError("Patient labels must be control=0 or ASD=1")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Patient probabilities must be finite and between zero and one")
    return labels.astype(int), probabilities


def patient_metrics(y, probabilities, threshold=0.5):
    labels, probabilities = _inputs(y, probabilities)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid frozen probability threshold")
    predicted = probabilities >= threshold
    tp = int(((labels == 1) & predicted).sum())
    tn = int(((labels == 0) & ~predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    n, n_asd, n_control = len(labels), tp + fn, tn + fp
    sensitivity = tp / n_asd if n_asd else None
    specificity = tn / n_control if n_control else None
    both = bool(n_asd and n_control)
    warnings = []
    if not tp + fp:
        warnings.append("No positive predictions; precision is defined as zero.")
    if not both:
        warnings.append("One-class/empty cohort: class-conditional/AUC metrics may be undefined.")
    return {
        "n_patients": n, "n_asd": n_asd, "n_control": n_control,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy": (tp + tn) / n if n else None,
        "balanced_accuracy": (sensitivity + specificity) / 2 if both else None,
        "precision": tp / (tp + fp) if tp + fp else 0.0 if n else None,
        "recall": sensitivity, "sensitivity": sensitivity, "specificity": specificity,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0 if n else None,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both else None,
        "average_precision": float(average_precision_score(labels, probabilities)) if both else None,
        "threshold": float(threshold), "warnings": warnings,
    }


def best_threshold(y, probabilities):
    labels, probabilities = _inputs(y, probabilities)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Threshold selection requires both training diagnosis classes")
    scores = [(patient_metrics(labels, probabilities, tick / 100)["balanced_accuracy"], tick / 100)
              for tick in range(5, 96)]
    best = max(score for score, _ in scores)
    tied = [threshold for score, threshold in scores if abs(score - best) <= 1e-12]
    return min(tied, key=lambda threshold: (round(abs(threshold - 0.5), 12), threshold))


def wilson_interval(successes, n, z=1.959963984540054):
    if n <= 0:
        return [None, None]
    if not 0 <= successes <= n:
        raise ValueError("Invalid binomial success count")
    p, denominator = successes / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def bootstrap_intervals(y, probabilities, threshold=0.5, n_bootstrap=2000, seed=42):
    labels, probabilities = _inputs(y, probabilities)
    if set(labels.tolist()) != {0, 1} or n_bootstrap < 1:
        raise ValueError("Patient bootstrap requires two classes and a positive sample count")
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(labels == value) for value in (0, 1)]
    keys = ("balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc", "average_precision")
    samples = {key: [] for key in keys}
    for _ in range(n_bootstrap):
        ids = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
        measured = patient_metrics(labels[ids], probabilities[ids], threshold)
        for key in keys:
            samples[key].append(measured[key])
    measured = patient_metrics(labels, probabilities, threshold)
    lower, upper = wilson_interval(measured["tp"] + measured["tn"], len(labels))
    output = {"accuracy": {"lower": lower, "upper": upper, "method": "95% Wilson"},
              "bootstrap_seed": seed, "bootstrap_samples": n_bootstrap}
    for key, values in samples.items():
        lower, upper = np.percentile(values, [2.5, 97.5])
        output[key] = {
            "lower": float(lower), "upper": float(upper),
            "method": "95% class-stratified patient percentile bootstrap",
        }
    return output


def stored_curves(y, probabilities):
    labels, probabilities = _inputs(y, probabilities)
    if set(labels.tolist()) != {0, 1}:
        return {}
    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return {"roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr": {"recall": recall.tolist(), "precision": precision.tolist()}}


def _subgroups(rows, probabilities, threshold):
    return {}


def evaluate_frozen(run_dir, settings, test_rows, splits):
    import os
    from contextlib import contextmanager
    from pathlib import Path

    import joblib

    from .artifacts import atomic_csv, atomic_json, ensure_safe_output, read_json, sha256_file, stable_hash
    from .config import config_hash
    from .dataset import verify_sources
    from .deep import FrozenEncoder
    from .features import FAMILIES, extract_features
    from .training import _probability

    @contextmanager
    def locked(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    run_dir = ensure_safe_output(Path(run_dir), settings)
    if settings.purpose != "production" or splits.get("purpose") != "production":
        raise ValueError("Smoke runs cannot consume a production test evaluation")
    ids = [row["subject_id"] for row in test_rows]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(splits["test_ids"]):
        raise ValueError("Final scoring requires every locked test patient exactly once")
    if set(ids) & set(splits["development_ids"]):
        raise ValueError("Test/development patient overlap")
    if stable_hash({k: v for k, v in splits.items() if k != "split_hash"}) != splits["split_hash"]:
        raise ValueError("Split membership changed after preparation")
    manifest = read_json(run_dir / "frozen_bundle_manifest.json")
    frozen_hash = manifest.get("freeze_hash")
    if frozen_hash != stable_hash({k: v for k, v in manifest.items() if k != "freeze_hash"}):
        raise ValueError("Frozen model manifest fingerprint mismatch")
    if manifest["config_hash"] != config_hash(settings) or manifest["split_hash"] != splits["split_hash"]:
        raise ValueError("Frozen configuration/split changed")
    if manifest.get("purpose") != "production" or set(manifest["bundles"]) != set(FAMILIES):
        raise ValueError("Incomplete or nonproduction frozen model family set")
    if manifest["development_ids"] != splits["development_ids"]:
        raise ValueError("Frozen training membership differs from the split")
    for bundle in manifest["bundles"].values():
        if set(bundle["fit_ids"]) != set(splits["development_ids"]):
            raise ValueError("A frozen model did not fit exclusively on development patients")
        if sha256_file(run_dir / bundle["file"]) != bundle["sha256"]:
            raise ValueError("Frozen Random Forest bundle changed")
    weights = run_dir / manifest["weights"]["file"]
    if sha256_file(weights) != manifest["weights"]["sha256"] or (
        manifest["weights"]["sha256"] != manifest["encoder"]["weight_sha256"]
    ):
        raise ValueError("Frozen EfficientNet weights changed")
    verify_sources(test_rows)
    registry = ensure_safe_output(
        settings.repo_root / "runs" / "hybrid" / "protocols" / splits["cohort_hash"], settings,
    )
    registered_splits = read_json(registry / "splits.json")
    if registered_splits != splits:
        raise ValueError("Immutable patient split registry differs from this run")
    claim_path = registry / "primary_test_claim.json"
    with locked(registry / "evaluation.lock"):
        claim = read_json(claim_path) if claim_path.is_file() else None
        if claim and (claim.get("freeze_hash") != frozen_hash or claim.get("run_dir") != str(run_dir.resolve())):
            raise ValueError("This cohort's primary test is already claimed by a different frozen model/run")
        if claim and claim.get("status") == "complete":
            if sha256_file(run_dir / "test" / "metrics.json") != claim["metrics_sha256"]:
                raise ValueError("Saved final test evidence changed")
            return read_json(run_dir / "test" / "metrics.json")
        atomic_json(claim_path, {
            "freeze_hash": frozen_hash, "run_dir": str(run_dir.resolve()),
            "split_hash": splits["split_hash"], "status": "in_progress",
        })
        if manifest["encoder"].get("kind") == "finetuned":
            from .features import extract_deep_cached
            from .finetune import FineTunedEncoder

            encoder = FineTunedEncoder.load(
                run_dir / manifest["encoder"]["checkpoint"], settings)
            try:
                static, _, _ = extract_features(
                    test_rows, settings, run_dir / "cache" / "test_features",
                    backbones=(), allow_download=False,
                )
                deep = extract_deep_cached(
                    test_rows, settings, run_dir / "cache" / "test_features", encoder)
                frame = static.join(deep)
            finally:
                encoder.close()
        else:
            encoder = FrozenEncoder(
                manifest["backbone"], settings, allow_download=False, weights_path=weights,
            )
            try:
                current = encoder.metadata()
                if {k: v for k, v in current.items() if k != "weights_path"} != {
                    k: v for k, v in manifest["encoder"].items() if k != "weights_path"
                }:
                    raise ValueError("Frozen encoder numerical/feature recipe changed")
                frame, _, _ = extract_features(
                    test_rows, settings, run_dir / "cache" / "test_features",
                    backbones=(manifest["backbone"],), encoders={manifest["backbone"]: encoder},
                    allow_download=False,
                )
            finally:
                encoder.close()
        labels = np.array([row["label"] for row in test_rows], dtype=int)
        models, intervals, curves, all_predictions, primary_probabilities = {}, {}, {}, [], None
        for family in FAMILIES:
            saved = manifest["bundles"][family]
            bundle = joblib.load(run_dir / saved["file"])
            if (bundle["fit_ids"] != splits["development_ids"]
                    or bundle["transform"].fit_ids_ != splits["development_ids"]
                    or bundle["threshold"] != saved["threshold"]
                    or bundle["backbone"] != manifest["backbone"]
                    or bundle["family"] != family or bundle["config_hash"] != config_hash(settings)):
                raise ValueError("Frozen fitted-object provenance mismatch")
            matrix = bundle["transform"].transform(frame.loc[ids])
            if bundle["forest"].classes_.tolist() != [0, 1]:
                raise ValueError("Frozen RF diagnosis class mapping changed")
            probabilities = _probability(bundle["forest"], matrix)
            threshold = bundle["threshold"]
            result_path = run_dir / "test" / f"{family}_evidence.json"
            if result_path.is_file():
                previous = read_json(result_path)
                if previous["freeze_hash"] != frozen_hash or not np.allclose(
                    previous["probabilities"], probabilities, atol=0, rtol=0,
                ):
                    raise ValueError("Interrupted test evidence differs from this frozen prediction")
                evidence = previous
            else:
                evidence = {
                    "freeze_hash": frozen_hash, "probabilities": probabilities.tolist(),
                    "metrics": patient_metrics(labels, probabilities, threshold),
                    "intervals": bootstrap_intervals(labels, probabilities, threshold, seed=settings.seed),
                    "curves": stored_curves(labels, probabilities),
                }
                atomic_json(result_path, evidence)
            models[family], intervals[family], curves[family] = (
                evidence["metrics"], evidence["intervals"], evidence["curves"],
            )
            all_predictions.extend({
                "subject_id": sid, "family": family, "label": int(labels[i]),
                "probability": float(probabilities[i]), "threshold": threshold,
                "prediction": int(probabilities[i] >= threshold),
            } for i, sid in enumerate(ids))
            if family == "combined":
                primary_probabilities = probabilities
            print(f"locked test {family}: accuracy={models[family]['accuracy']:.4f}", flush=True)
        summary = {
            "primary_model": "combined", "n_patients": len(ids), "models": models,
            "target_met": bool(models["combined"]["accuracy"] > 0.8),
            "majority_class_accuracy": float(max((labels == 0).mean(), (labels == 1).mean())),
            "intervals": intervals, "curves": curves,
            "subgroups": _subgroups(test_rows, primary_probabilities, manifest["bundles"]["combined"]["threshold"]),
            "freeze_hash": frozen_hash, "split_hash": splits["split_hash"],
        }
        atomic_csv(run_dir / "test" / "predictions.csv", all_predictions)
        atomic_json(run_dir / "test" / "intervals.json", intervals)
        atomic_json(run_dir / "test" / "metrics.json", summary)
        report = read_json(run_dir / "report_data.json")
        report.update({"status": "complete", "test": summary})
        atomic_json(run_dir / "report_data.json", report)
        atomic_json(claim_path, {
            "freeze_hash": frozen_hash, "run_dir": str(run_dir.resolve()),
            "split_hash": splits["split_hash"], "status": "complete",
            "metrics_sha256": sha256_file(run_dir / "test" / "metrics.json"),
        })
        return summary
