"""Paper-style slice-level evaluation.

Reproduces the reference paper's protocol: slices (not patients) are assigned
to stratified cross-validation folds, so slices of the same brain appear in
both the training and evaluation partition of every fold. This intentionally
mirrors the paper's design so its reported accuracy is comparable; it is NOT
a patient-level performance estimate.
"""
import csv as _csv
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from .artifacts import atomic_csv, atomic_json, read_json, sha256_file, stable_hash
from .features import _feature_recipe
from .finetune import FineTunedEncoder
from .preprocessing import prepare_subject
from .saliency import enhance_slice, saliency_stats
from .texture import slice_texture

FAMILIES = ("cnn_only", "texture_only", "saliency_only", "deep_only", "mri_hybrid", "combined")
RF_BLOCKS = {
    "texture_only": ("texture",),
    "saliency_only": ("saliency",),
    "deep_only": ("deep",),
    "mri_hybrid": ("texture", "deep"),
    "combined": ("texture", "saliency", "deep"),
}


def _metrics(y_true, prob):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "sensitivity": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y_true)),
    }
    metrics["auroc"] = (
        float(roc_auc_score(y_true, prob)) if len(set(y_true.tolist())) == 2 else None
    )
    return metrics


def _slice_cache(row, settings, cache_dir):
    expected = {
        "subject_id": str(row["subject_id"]),
        "source_sha256": row["source_sha256"],
        "recipe": _feature_recipe(settings),
    }
    key = stable_hash(expected)
    stem = f"slice_{expected['subject_id']}_{key}"
    return Path(cache_dir) / f"{stem}.npz", Path(cache_dir) / f"{stem}.json", expected


def _slice_patient(task):
    row, settings, cache_dir, deadline = task
    if time.monotonic() >= deadline:
        raise TimeoutError("Soft budget reached; completed slice caches are preserved")
    sid = str(row["subject_id"])
    if "source_size" in row or "source_mtime_ns" in row:
        stat = Path(row["raw_path"]).stat()
        if ("source_size" in row and stat.st_size != row["source_size"]) or (
            "source_mtime_ns" in row and stat.st_mtime_ns != row["source_mtime_ns"]
        ):
            raise ValueError(f"Source scan changed during extraction: {sid}")
    path, meta_path, expected = _slice_cache(row, settings, cache_dir)
    if path.is_file() and meta_path.is_file():
        saved = read_json(meta_path)
        if saved["input"] != expected or sha256_file(path) != saved.get("output_sha256"):
            raise ValueError(f"Slice cache provenance mismatch for {sid}")
        return sid, str(path)
    prepared = prepare_subject(row["raw_path"], settings)
    payload = {}
    for view in settings.views:
        texture_rows, saliency_rows, grays = [], [], []
        for gray, mask in prepared["views"][view]:
            vec, _ = slice_texture(gray, mask, settings)
            enhanced, sal_map = enhance_slice(gray, settings.saliency_alpha)
            texture_rows.append(np.asarray(vec, dtype=np.float32))
            saliency_rows.append(np.asarray(saliency_stats(sal_map), dtype=np.float32))
            grays.append(np.asarray(enhanced, dtype=np.float16))
        payload[f"texture_{view}"] = np.stack(texture_rows)
        payload[f"saliency_{view}"] = np.stack(saliency_rows)
        payload[f"cnn_{view}"] = np.stack(grays)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)
    atomic_json(meta_path, {
        "input": expected, "qc": prepared["qc"], "output_sha256": sha256_file(path),
    })
    return sid, str(path)


def _prewarm_stripper():
    """Trigger the hd-bet weight download once, serially, before worker processes race on it."""
    if not shutil.which("hd-bet"):
        return
    with tempfile.TemporaryDirectory(prefix="hdbet_warm_") as tmp:
        source = os.path.join(tmp, "warm.nii.gz")
        output = os.path.join(tmp, "warm_bet.nii.gz")
        rng = np.random.default_rng(0)
        volume = (rng.random((48, 48, 48), dtype=np.float32) * 0.2 + 0.4)
        volume[:8] = volume[-8:] = volume[:, :8] = volume[:, -8:] = volume[:, :, :8] = volume[:, :, -8:] = 0
        nib.save(nib.Nifti1Image(volume, np.eye(4)), source)
        try:
            subprocess.run(
                ["hd-bet", "-i", source, "-o", output, "--disable_tta"],
                check=True, capture_output=True, timeout=900,
            )
        except (OSError, subprocess.SubprocessError) as error:
            print(f"warning: hd-bet warm-up failed ({error}); workers will use morphology", flush=True)


def extract_slices(rows, settings, cache_dir, workers=0, deadline=None):
    """Per-slice feature extraction (texture + saliency + enhanced CNN slices).

    Patients that fail preprocessing QC are recorded in ``failures`` and dropped;
    the evaluation proceeds with the remainder.
    """
    deadline = deadline if deadline is not None else time.monotonic() + settings.soft_budget_hours * 3600
    workers = workers or min(8, os.cpu_count() or 4)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = {str(row["subject_id"]): (row, settings, str(cache_dir), deadline) for row in rows}
    paths, failures = {}, {}
    if workers <= 1 or len(tasks) < 4:
        for sid, task in tasks.items():
            try:
                _, path = _slice_patient(task)
                paths[sid] = path
            except (ValueError, OSError) as error:
                failures[sid] = str(error)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            futures = {pool.submit(_slice_patient, task): sid for sid, task in tasks.items()}
            for index, future in enumerate(as_completed(futures), 1):
                sid = futures[future]
                try:
                    _, path = future.result()
                    paths[sid] = path
                except (ValueError, OSError) as error:
                    failures[sid] = str(error)
                if index == 1 or index % 25 == 0 or index == len(tasks):
                    print(f"slices: {index}/{len(tasks)} patients extracted "
                          f"({len(failures)} failed)", flush=True)
    for sid, error in failures.items():
        print(f"  warning: patient {sid} dropped during extraction: {error}", flush=True)
    if not paths:
        raise ValueError("Slice extraction produced no usable patients")
    return paths, failures


def _augment_gray(gray, rng):
    """Small geometric/intensity jitter producing a near-duplicate of the slice."""
    import cv2
    gray = np.asarray(gray, dtype=np.float32)
    angle = float(rng.uniform(-8.0, 8.0))
    scale = float(rng.uniform(0.96, 1.04))
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    aug = cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    if rng.random() < 0.5:
        aug = np.ascontiguousarray(aug[:, ::-1])
    gain = float(rng.uniform(0.92, 1.08))
    return np.clip(aug * gain, 0.0, 1.0).astype(np.float32)


def _assemble(rows, settings, paths, aug_copies=0):
    """Materialize per-patient slice arrays and a flat slice index.

    With ``aug_copies > 0`` each slice also yields near-duplicate augmented
    samples; they enter the fold pool as independent items, so twins of a
    training slice can land in the evaluation partition (paper-style protocol).
    Texture/saliency vectors of augmented samples reuse the source slice's
    features; only the CNN input is re-rendered.
    """
    slices, arrays = [], {}
    rng = np.random.default_rng(settings.seed)
    for row in sorted(rows, key=lambda r: str(r["subject_id"])):
        sid = str(row["subject_id"])
        if sid not in paths:
            continue
        with np.load(paths[sid], allow_pickle=False) as bundle:
            arrays[sid] = {name: bundle[name].copy() for name in bundle.files}
        for view in settings.views:
            for index in range(arrays[sid][f"cnn_{view}"].shape[0]):
                slices.append({
                    "pid": sid, "view": view, "index": index,
                    "copy": 0, "label": int(row["label"]),
                })
        if aug_copies:
            enhanced_all = {}
            for view in settings.views:
                enhanced_all[view] = arrays[sid][f"cnn_{view}"].astype(np.float32)
                arrays[sid][f"cnn_aug_{view}"] = []
            for view in settings.views:
                for index in range(enhanced_all[view].shape[0]):
                    for copy in range(1, aug_copies + 1):
                        aug = _augment_gray(enhanced_all[view][index], rng)
                        arrays[sid][f"cnn_aug_{view}"].append(
                            (index, copy, aug.astype(np.float16)))
                        slices.append({
                            "pid": sid, "view": view, "index": index,
                            "copy": copy, "label": int(row["label"]),
                        })
            for view in settings.views:
                arrays[sid][f"cnn_aug_{view}"] = {
                    (i, c): g for i, c, g in arrays[sid][f"cnn_aug_{view}"]}
    if len({s["pid"] for s in slices}) != len(arrays):
        raise ValueError("Slice assembly dropped patients")
    return slices, arrays


def _gray(arrays, slices, i):
    s = slices[i]
    if s["copy"]:
        return np.asarray(
            arrays[s["pid"]][f"cnn_aug_{s['view']}"][(s["index"], s["copy"])],
            dtype=np.float32)
    return np.asarray(arrays[s["pid"]][f"cnn_{s['view']}"][s["index"]], dtype=np.float32)


def _block(arrays, slices, indices, block):
    rows = []
    for i in indices:
        s = slices[i]
        rows.append(arrays[s["pid"]][f"{block}_{s['view']}"][s["index"]])
    return np.stack(rows).astype(np.float32)


def _run_rf(name, blocks, arrays, slices, train_idx, test_idx, emb_train, emb_test, y_train, settings):
    feats_train, feats_test = {}, {}
    for block in blocks:
        if block == "deep":
            feats_train[block], feats_test[block] = emb_train, emb_test
        else:
            feats_train[block] = _block(arrays, slices, train_idx, block)
            feats_test[block] = _block(arrays, slices, test_idx, block)
    x_train = np.concatenate([feats_train[b] for b in blocks], axis=1)
    x_test = np.concatenate([feats_test[b] for b in blocks], axis=1)
    forest = RandomForestClassifier(
        n_estimators=settings.n_estimators, n_jobs=settings.rf_jobs,
        random_state=settings.seed, class_weight="balanced_subsample",
    )
    forest.fit(x_train, y_train)
    return forest.predict_proba(x_test)[:, 1].astype(np.float32)


def _pseudo_patients(arrays, slices, indices, settings):
    """Wrap each slice as its own 'patient' so the internal split is slice-level."""
    patients, labels = {}, {}
    for i in indices:
        s = slices[i]
        pseudo = f"{s['pid']}__{s['view']}__{s['index']}__{s['copy']}"
        views = {v: np.empty((0, 1, 1), np.float32) for v in settings.views}
        views[s["view"]] = _gray(arrays, slices, i)[None]
        patients[pseudo] = (pseudo, views)
        labels[pseudo] = int(s["label"])
    return list(patients.values()), labels


def run(run_dir, settings, rows, aug_copies=0):
    """Slice-level 5-fold CV: end-to-end B5 classifier + per-slice RF families."""
    run_dir = Path(run_dir)
    folds_dir = run_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + settings.soft_budget_hours * 3600
    if shutil.which("hd-bet"):
        print("hd-bet detected; warming up skull-strip weights", flush=True)
        _prewarm_stripper()
    paths, failures = extract_slices(rows, settings, run_dir / "slice_cache",
                                     settings.extract_workers, deadline)
    slices, arrays = _assemble(rows, settings, paths, aug_copies)
    y = np.asarray([s["label"] for s in slices], dtype=int)
    print(f"paper-style eval: {len(arrays)} patients -> {len(slices)} slices "
          f"({int((y == 1).sum())} ASD / {int((y == 0).sum())} control)", flush=True)
    backbone = (settings.finetune or {}).get("backbone", "efficientnet_b5")
    kfold = StratifiedKFold(
        n_splits=settings.outer_folds, shuffle=True, random_state=settings.seed)
    pooled = {fam: {"y": [], "p": [], "pid": []} for fam in FAMILIES}
    fold_summaries = []
    for fold, (train_idx, test_idx) in enumerate(
            kfold.split(np.zeros(len(y)), y), 1):
        fold_json = folds_dir / f"fold{fold}.json"
        pred_csv = folds_dir / f"fold{fold}_predictions.csv"
        if fold_json.is_file() and pred_csv.is_file():
            summary = read_json(fold_json)
            fold_summaries.append(summary)
            with pred_csv.open(newline="", encoding="utf-8") as handle:
                for record in _csv.DictReader(handle):
                    pooled[record["family"]]["y"].append(int(record["y_true"]))
                    pooled[record["family"]]["p"].append(float(record["prob"]))
                    pooled[record["family"]]["pid"].append(record["pid"])
            print(f"fold {fold}/{settings.outer_folds}: loaded previous result", flush=True)
            continue
        if time.monotonic() >= deadline:
            raise TimeoutError("Soft budget reached; completed folds are preserved")
        print(f"fold {fold}/{settings.outer_folds}: {len(train_idx)} train / "
              f"{len(test_idx)} test slices", flush=True)
        y_train = y[train_idx]
        encoder_path = folds_dir / f"fold{fold}" / "encoder.pt"
        if encoder_path.is_file() and encoder_path.with_suffix(".json").is_file():
            encoder = FineTunedEncoder.load(encoder_path, settings)
            print(f"fold {fold}: loaded fine-tuned {backbone} encoder", flush=True)
        else:
            patients, labels = _pseudo_patients(arrays, slices, train_idx, settings)
            encoder = FineTunedEncoder(settings, backbone)
            encoder.finetune(patients, labels, settings.finetune, encoder_path, deadline,
                             progress=lambda m: print(f"fold {fold}: {m}", flush=True))
        train_grays = [_gray(arrays, slices, i) for i in train_idx]
        test_grays = [_gray(arrays, slices, i) for i in test_idx]
        emb_train = encoder._embed_grays(train_grays)
        emb_test = encoder._embed_grays(test_grays)
        probs = {"cnn_only": encoder.slice_probabilities(test_grays)}
        encoder.close()
        del encoder, train_grays, test_grays
        for family in FAMILIES[1:]:
            probs[family] = _run_rf(
                family, RF_BLOCKS[family], arrays, slices, train_idx, test_idx,
                emb_train, emb_test, y_train, settings)
        summary = {"fold": fold, "n_train": int(len(train_idx)),
                   "n_test": int(len(test_idx)), "families": {}}
        pred_rows = []
        for family in FAMILIES:
            summary["families"][family] = _metrics(y[test_idx], probs[family])
            for i, prob in zip(test_idx, probs[family]):
                s = slices[i]
                pooled[family]["y"].append(int(s["label"]))
                pooled[family]["p"].append(float(prob))
                pooled[family]["pid"].append(s["pid"])
                pred_rows.append({
                    "fold": fold, "family": family, "pid": s["pid"], "view": s["view"],
                    "slice_index": s["index"], "y_true": int(s["label"]),
                    "prob": float(prob),
                })
        atomic_json(fold_json, summary)
        atomic_csv(pred_csv, pred_rows)
        fold_summaries.append(summary)
        accs = {f: summary["families"][f]["accuracy"] for f in FAMILIES}
        print(f"fold {fold} done: " + " ".join(f"{f}={a:.3f}" for f, a in accs.items()),
              flush=True)
        del emb_train, emb_test, probs

    families = {}
    for family in FAMILIES:
        fold_accs = [s["families"][family]["accuracy"] for s in fold_summaries]
        pooled_m = _metrics(pooled[family]["y"], pooled[family]["p"])
        # subject-level majority vote over that patient's test slices
        votes = {}
        for pid, prob, truth in zip(pooled[family]["pid"], pooled[family]["p"],
                                    pooled[family]["y"]):
            votes.setdefault(pid, {"p": [], "y": truth})["p"].append(prob)
        vote_y = np.asarray([v["y"] for v in votes.values()])
        vote_p = np.asarray([float(np.mean(v["p"])) for v in votes.values()])
        families[family] = {
            "fold_accuracies": fold_accs,
            "mean_accuracy": float(np.mean(fold_accs)),
            "std_accuracy": float(np.std(fold_accs)),
            "pooled_slice_metrics": pooled_m,
            "subject_vote_metrics": _metrics(vote_y, vote_p),
        }
    result = {
        "protocol": "slice_level_stratified_5fold_cv",
        "warning": ("Slices of the same patient can appear in both train and test "
                    "partitions (paper-style evaluation); accuracies are inflated "
                    "relative to patient-disjoint evaluation."),
        "n_patients": len(arrays), "n_slices": int(len(y)),
        "extraction_failures": failures,
        "backbone": backbone, "families": families,
    }
    atomic_json(run_dir / "metrics.json", result)
    print("\n=== paper-style slice-level CV results ===", flush=True)
    for family in FAMILIES:
        f = families[family]
        print(f"{family:>14}: slice acc {f['mean_accuracy']:.3f} "
              f"(pooled {f['pooled_slice_metrics']['accuracy']:.3f}) | "
              f"subject vote {f['subject_vote_metrics']['accuracy']:.3f}", flush=True)
    return result
