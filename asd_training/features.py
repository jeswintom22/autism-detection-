import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import atomic_json, read_json, sha256_file, stable_hash
from .deep import FrozenEncoder
from .preprocessing import prepare_subject
from .saliency import extract_saliency
from .texture import extract_texture


FAMILIES = ("texture_only", "saliency_only", "deep_only", "mri_hybrid", "combined")
BLOCKS = {
    "deep_only": ("deep",),
    "texture_only": ("texture",),
    "saliency_only": ("saliency",),
    "mri_hybrid": ("deep", "texture"),
    "combined": ("deep", "texture", "saliency"),
}


def deep_names_for(backbone, dimension, views):
    return [
        f"deep__{backbone}__{view}__{statistic}__{i:04d}"
        for view in views for statistic in ("mean", "std") for i in range(2 * dimension)
    ]


RECIPE_KEYS = (
    "preprocessing_version", "schema_version", "n_slices", "spacing", "margin_mm",
    "percentiles", "views", "glcm_levels", "glcm_distances", "glcm_angles",
    "glcm_properties", "lbp_points", "lbp_radii", "lbp_bins",
    "nimble_window", "nimble_g", "canny_low", "canny_high",
    "saliency_alpha",
)


def _feature_recipe(settings, metadata=None):
    recipe = {key: settings.to_dict()[key] for key in RECIPE_KEYS}
    recipe["skull_strip_tool"] = next(
        (tool for tool in ("bet", "hd-bet") if shutil.which(tool)), "morphology")
    recipe["encoders"] = {
        name: {k: v for k, v in info.items() if k != "weights_path"}
        for name, info in (metadata or {}).items()
    }
    return recipe


def _patient_cache(row, settings, cache_dir, metadata=None):
    expected = {
        "subject_id": str(row["subject_id"]),
        "source_sha256": row["source_sha256"],
        "recipe": _feature_recipe(settings, metadata),
    }
    key = stable_hash(expected)
    return (Path(cache_dir) / f"{expected['subject_id']}_{key}.npz",
            Path(cache_dir) / f"{expected['subject_id']}_{key}.json", expected)


def load_patient_cnn(row, settings, cache_dir):
    """Return {view: (n_slices, H, W) float32 saliency-enhanced slices} from the patient cache."""
    path, meta_path, expected = _patient_cache(row, settings, cache_dir)
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Patient CNN slice cache missing for {expected['subject_id']}")
    saved = read_json(meta_path)
    if saved["input"] != expected:
        raise ValueError("Patient feature cache provenance mismatch")
    if sha256_file(path) != saved.get("output_sha256"):
        raise ValueError("Patient feature cache content fingerprint mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        return {
            view: arrays[f"cnn_{view}"].astype(np.float32) for view in settings.views
        }


def deep_names_cached(backbone, embedding_dim, views):
    return [
        f"deep__{backbone}__{view}__{statistic}__{i:04d}"
        for view in views for statistic in ("mean", "std") for i in range(embedding_dim)
    ]


def extract_deep_cached(rows, settings, cache_dir, encoder):
    """Embed cached CNN slices for patients; returns a patient-indexed deep-feature frame."""
    records, ids = [], []
    for index, row in enumerate(rows, 1):
        cnn_views = load_patient_cnn(row, settings, cache_dir)
        vector = encoder.subject_features(cnn_views)
        records.append(vector)
        ids.append(str(row["subject_id"]))
        if index == 1 or index % 50 == 0 or index == len(rows):
            print(f"deep features: {index}/{len(rows)} patients ({encoder.name})", flush=True)
    columns = deep_names_cached(encoder.name, encoder.embedding_dim, settings.views)
    frame = pd.DataFrame(np.stack(records).astype(np.float32),
                         index=pd.Index(ids, name="subject_id"), columns=columns)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Nonfinite fine-tuned patient embeddings")
    return frame


def _static_patient(row, settings, cache_dir, deadline):
    """Texture + saliency + cached CNN slices for one patient (no deep encoder)."""
    if time.monotonic() >= deadline:
        raise TimeoutError("Soft budget reached; completed patient feature caches are preserved")
    sid = str(row["subject_id"])
    if "source_size" in row or "source_mtime_ns" in row:
        source_stat = Path(row["raw_path"]).stat()
        if ("source_size" in row and source_stat.st_size != row["source_size"]) or (
            "source_mtime_ns" in row and source_stat.st_mtime_ns != row["source_mtime_ns"]
        ):
            raise ValueError(f"Source scan changed during extraction: {sid}")
    path, meta_path, expected = _patient_cache(row, settings, cache_dir)
    if path.is_file() and meta_path.is_file():
        saved = read_json(meta_path)
        if saved["input"] != expected:
            raise ValueError("Patient feature cache provenance mismatch")
        if sha256_file(path) != saved.get("output_sha256"):
            raise ValueError("Patient feature cache content fingerprint mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            texture, saliency = arrays["texture"].copy(), arrays["saliency"].copy()
        texture_names, saliency_names = saved["texture_names"], saved["saliency_names"]
    else:
        prepared = prepare_subject(row["raw_path"], settings)
        texture, texture_names = extract_texture(prepared, settings)
        saliency, saliency_names, prepared = extract_saliency(prepared, settings)
        cnn_payload = {
            f"cnn_{view}": np.stack(
                [gray for gray, _ in prepared["cnn_views"][view]]
            ).astype(np.float16)
            for view in settings.views
        }
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, texture=texture, saliency=saliency, **cnn_payload)
        os.replace(temporary, path)
        atomic_json(meta_path, {
            "input": expected, "texture_names": texture_names,
            "saliency_names": saliency_names, "qc": prepared["qc"],
            "output_sha256": sha256_file(path),
        })
    return sid, texture, saliency, texture_names, saliency_names


def _static_worker(task):
    row, settings, cache_dir, deadline = task
    return _static_patient(row, settings, cache_dir, deadline)


def extract_static_parallel(rows, settings, cache_dir, workers=0, deadline=None):
    """Patient-parallel static features (texture + saliency + CNN slice cache)."""
    deadline = deadline if deadline is not None else time.monotonic() + settings.soft_budget_hours * 3600
    if not rows:
        raise ValueError("No patients for feature extraction")
    workers = workers or min(8, os.cpu_count() or 4)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if workers <= 1 or len(rows) < 4:
        results = [_static_patient(row, settings, cache_dir, deadline) for row in rows]
    else:
        tasks = [(row, settings, str(cache_dir), deadline) for row in rows]
        results = []
        with ProcessPoolExecutor(max_workers=min(workers, len(rows))) as pool:
            futures = {pool.submit(_static_worker, task): task[0] for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if index == 1 or index % 25 == 0 or index == len(rows):
                    print(f"features: {index}/{len(rows)} patients (static)", flush=True)
    ids = [result[0] for result in results]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate patient feature row")
    texture_names = results[0][3]
    saliency_names = results[0][4]
    for _, _, _, tn, sn in results[1:]:
        if tn != texture_names or sn != saliency_names:
            raise ValueError("Patient feature schema drift")
    records = [
        np.concatenate([texture, saliency]).astype(np.float32)
        for _, texture, saliency, _, _ in results
    ]
    columns = list(texture_names) + list(saliency_names)
    frame = pd.DataFrame(np.stack(records), index=pd.Index(ids, name="subject_id"), columns=columns)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Nonfinite MRI features")
    schema = {"columns": columns, "recipe": _feature_recipe(settings),
              "schema_hash": stable_hash(columns)}
    return frame, schema


def extract_features(rows, settings, cache_dir, backbones=None, encoders=None, allow_download=True,
                     deadline=None):
    deadline = deadline if deadline is not None else time.monotonic() + settings.soft_budget_hours * 3600
    backbones = tuple(settings.backbones if backbones is None else backbones)
    owned = encoders is None
    encoders = encoders if encoders is not None else {
        name: FrozenEncoder(name, settings, allow_download=allow_download) for name in backbones
    }
    metadata = {name: encoders[name].metadata() for name in backbones}
    recipe = _feature_recipe(settings, metadata)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    records, ids, texture_names, saliency_names = [], [], None, None
    try:
        for index, row in enumerate(rows, 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("Soft budget reached; completed patient feature caches are preserved")
            sid = str(row["subject_id"])
            if "source_size" in row or "source_mtime_ns" in row:
                source_stat = Path(row["raw_path"]).stat()
                if ("source_size" in row and source_stat.st_size != row["source_size"]) or (
                    "source_mtime_ns" in row and source_stat.st_mtime_ns != row["source_mtime_ns"]
                ):
                    raise ValueError(f"Source scan changed during extraction: {sid}")
            if sid in ids:
                raise ValueError(f"Duplicate patient feature row: {sid}")
            expected = {"subject_id": sid, "source_sha256": row["source_sha256"], "recipe": recipe}
            key = stable_hash(expected)
            path, meta_path = cache_dir / f"{sid}_{key}.npz", cache_dir / f"{sid}_{key}.json"
            if path.is_file() and meta_path.is_file():
                saved = read_json(meta_path)
                if saved["input"] != expected:
                    raise ValueError("Patient feature cache provenance mismatch")
                if sha256_file(path) != saved.get("output_sha256"):
                    raise ValueError("Patient feature cache content fingerprint mismatch")
                with np.load(path, allow_pickle=False) as arrays:
                    texture = arrays["texture"].copy()
                    saliency = arrays["saliency"].copy()
                    vectors = {name: arrays[name].copy() for name in backbones}
                current_texture_names = saved["texture_names"]
                current_saliency_names = saved["saliency_names"]
            else:
                prepared = prepare_subject(row["raw_path"], settings)
                texture, current_texture_names = extract_texture(prepared, settings)
                saliency, current_saliency_names, prepared = extract_saliency(prepared, settings)
                vectors = {name: encoders[name].subject_features(prepared) for name in backbones}
                cnn_payload = {
                    f"cnn_{view}": np.stack(
                        [gray for gray, _ in prepared["cnn_views"][view]]
                    ).astype(np.float16)
                    for view in settings.views
                }
                temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, texture=texture, saliency=saliency,
                                        **cnn_payload, **vectors)
                os.replace(temporary, path)
                atomic_json(meta_path, {
                    "input": expected, "texture_names": current_texture_names,
                    "saliency_names": current_saliency_names, "qc": prepared["qc"],
                    "output_sha256": sha256_file(path),
                })
                del prepared
            if texture_names is not None and texture_names != current_texture_names:
                raise ValueError("Patient texture schema drift")
            if saliency_names is not None and saliency_names != current_saliency_names:
                raise ValueError("Patient saliency schema drift")
            texture_names = current_texture_names
            saliency_names = current_saliency_names
            expected_deep = [4 * metadata[name]["dimension"] * len(settings.views) for name in backbones]
            if texture.shape != (72 * len(settings.views),) or saliency.shape != (12 * len(settings.views),) or any(
                vectors[name].shape != (dimension,)
                for name, dimension in zip(backbones, expected_deep)
            ):
                raise ValueError("Cached patient feature dimensions do not match the schema")
            raw = np.concatenate(
                [vectors[name] for name in backbones] + [texture, saliency]
            )
            if not np.isfinite(raw).all():
                raise ValueError(f"Nonfinite MRI features for patient {sid}")
            records.append(raw.astype(np.float32))
            ids.append(sid)
            if index == 1 or index % 25 == 0 or index == len(rows):
                print(f"features: {index}/{len(rows)} patients ({', '.join(backbones)})", flush=True)
    finally:
        if owned:
            for encoder in encoders.values():
                encoder.close()
    if not rows:
        raise ValueError("No patients for feature extraction")
    columns = [
        name for backbone in backbones
        for name in deep_names_for(backbone, metadata[backbone]["dimension"], settings.views)
    ]
    columns += texture_names + saliency_names
    frame = pd.DataFrame(np.stack(records), index=pd.Index(ids, name="subject_id"), columns=columns)
    schema = {"columns": columns, "recipe": recipe, "schema_hash": stable_hash(columns)}
    return frame, metadata, schema


class FeatureTransform:
    def __init__(self, settings, backbone, family):
        if family not in FAMILIES:
            raise ValueError(f"Unknown feature family: {family}")
        self.settings, self.backbone, self.family = settings, backbone, family

    def _check_frame(self, frame):
        if not isinstance(frame, pd.DataFrame) or not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError("Features require a unique, patient-indexed DataFrame")
        forbidden = [name for name in frame.columns if not isinstance(name, str) or not (
            name.startswith("deep__efficientnet_b0__") or name.startswith("deep__efficientnet_b5__")
            or name.startswith("texture__") or name.startswith("saliency__")
        )]
        if forbidden:
            raise ValueError(f"Non-allowlisted predictors: {forbidden}")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("MRI features must be finite")

    def fit(self, frame, y=None):
        self._check_frame(frame)
        self.fit_ids_ = [str(sid) for sid in frame.index]
        self.deep_columns_ = [name for name in frame if name.startswith(f"deep__{self.backbone}__")]
        texture = [name for name in frame if name.startswith("texture__")]
        saliency = [name for name in frame if name.startswith("saliency__")]
        blocks = BLOCKS[self.family]
        transforms = []
        if "deep" in blocks:
            if not self.deep_columns_ or len(frame) < 2:
                raise ValueError("Insufficient deep features or training patients for PCA")
            components = min(self.settings.pca_components, len(frame) - 1, len(self.deep_columns_))
            transforms.append(("deep", Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=components, svd_solver="randomized",
                            whiten=False, random_state=self.settings.seed)),
            ]), self.deep_columns_))
        if "texture" in blocks:
            if not texture:
                raise ValueError("Missing texture predictors")
            transforms.append(("texture", "passthrough", texture))
        if "saliency" in blocks:
            if not saliency:
                raise ValueError("Missing saliency predictors")
            transforms.append(("saliency", "passthrough", saliency))
        self.transformer_ = ColumnTransformer(transforms, remainder="drop", n_jobs=1)
        self.transformer_.fit(frame)
        self.required_columns_ = list(dict.fromkeys(
            column for _, _, columns in transforms for column in columns
        ))
        return self

    def transform(self, frame):
        self._check_frame(frame)
        if not hasattr(self, "transformer_"):
            raise ValueError("Feature transform is not fitted")
        if set(self.required_columns_) - set(frame):
            raise ValueError("Frozen feature schema is missing predictors")
        values = self.transformer_.transform(frame)
        values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite transformed predictors")
        return values

    def fit_transform(self, frame, y=None):
        return self.fit(frame, y).transform(frame)
