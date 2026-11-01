"""Step 2 - preprocess every scan in the manifest.

For each subject volume:
    1. skull-strip check - already stripped scans pass through;
       raw scans are stripped (HD-BET if installed, else a
       threshold + largest-component morphology fallback)
    2. adaptive "nimble" filter applied per axial slice:
           K(x,y) = g*f(x,y) + (1-g)*M(x,y) + T(x,y)*(f(x,y)-M(x,y))
       M = 4-neighbour mean (paper eq. 2), T = local Canny edge density
       damped by local noise. Fully vectorised - no per-window Python loops.
    3. saved to preprocessed_dataset/<class>/<id>_filtered.nii.gz
       (affine + header preserved, scl slope cleared)

Outputs step2_summary.json. Resumable: existing outputs are skipped.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import nibabel as nib
import numpy as np
import cv2
from joblib import Parallel, delayed
from scipy import ndimage

import config

_NEIGHBOUR4 = np.array([[0, 1, 0], [1, 0, 1], [1, 0, 1]], np.float32) / 4.0


# ------------------------------------------------------------ skull strip
def is_stripped(data):
    """Heuristic: stripped brains have modest nonzero fraction and an
    essentially empty outer rim of voxels."""
    nz_frac = float((data > 0).mean())
    s = config.STRIP_RIM_SLAB
    rim = np.concatenate([
        data[:s].ravel(), data[-s:].ravel(),
        data[:, :s].ravel(), data[:, -s:].ravel(),
        data[:, :, :s].ravel(), data[:, :, -s:].ravel(),
    ])
    rim_frac = float((rim > 0).mean())
    return nz_frac <= config.STRIP_MAX_NONZERO and rim_frac <= config.STRIP_MAX_RIM


def _strip_morphology(data):
    nz = data[data > 0]
    if nz.size == 0:
        return data
    mask = data > np.percentile(nz, 35)
    mask = ndimage.binary_opening(mask, iterations=2)
    lab, n = ndimage.label(mask)
    if n == 0:
        return data
    biggest = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    mask = lab == biggest
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_erosion(mask, iterations=1)
    return np.where(mask, data, 0).astype(data.dtype)


def _strip_hdbet(path):
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "brain.nii.gz")
        subprocess.run(["hd-bet", "-i", path, "-o", out],
                       check=True, capture_output=True)
        return np.asarray(nib.load(out).get_fdata(), dtype=np.float32)


def skull_strip(path, data):
    if shutil.which("hd-bet"):
        return _strip_hdbet(path), "hd-bet"
    return _strip_morphology(data), "morphology"


# ------------------------------------------------------------ nimble filter
def nimble_filter_slice(sl, window=config.WINDOW_SIZE, g=config.G_FILTER):
    """Vectorised adaptive nimble filter for one 2-D slice (returns 0-255 f32)."""
    f = cv2.normalize(sl.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    m = cv2.filter2D(f, cv2.CV_32F, _NEIGHBOUR4, borderType=cv2.BORDER_REPLICATE)
    edges = cv2.Canny(np.clip(f, 0, 255).astype(np.uint8),
                      config.CANNY_LOW, config.CANNY_HIGH)
    density = cv2.boxFilter((edges > 0).astype(np.float32),
                          cv2.CV_32F, (window, window))
    mean = cv2.boxFilter(f, cv2.CV_32F, (window, window))
    sq = cv2.boxFilter(f * f, cv2.CV_32F, (window, window))
    noise = np.sqrt(np.clip(sq - mean * mean, 0, None)) / 255.0
    t = np.clip(density * (1.0 - noise), 0.0, 1.0)
    return np.clip(g * f + (1.0 - g) * m + t * (f - m), 0, 255).astype(np.float32)


# ----------------------------------------------------------------- pipeline
def out_name(path):
    stem = os.path.basename(path)
    for suf in (".nii.gz", ".nii"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem.replace("_skull_stripped", "") + "_filtered.nii.gz"


def process_one(path, class_name, out_root):
    try:
        out_dir = os.path.join(out_root, class_name)
        out_path = os.path.join(out_dir, out_name(path))
        if os.path.exists(out_path):
            return {"path": path, "out": out_path, "method": "cached", "status": "ok"}
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)
        if is_stripped(data):
            method = "already_stripped"
        else:
            data, method = skull_strip(path, data)
        filt = np.stack(
            [nimble_filter_slice(data[:, :, z]) for z in range(data.shape[2])], axis=-1
        )
        out_img = nib.Nifti1Image(filt, img.affine, img.header.copy())
        hdr = out_img.header
        hdr["scl_slope"] = 1.0
        hdr["scl_inter"] = 0.0
        hdr["cal_max"] = 255.0
        hdr["cal_min"] = 0.0
        out_img.set_data_dtype(np.float32)
        nib.save(out_img, out_path)
        return {"path": path, "out": out_path, "method": method, "status": "ok"}
    except Exception as e:  # noqa: BLE001 - report and continue with other subjects
        return {"path": path, "status": "fail", "error": str(e)}


def read_manifest(manifest_path):
    with open(manifest_path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "ok"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=config.MANIFEST_PATH)
    ap.add_argument("--out-dir", default=config.PREPROCESSED_DIR)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--max", type=int, default=0, help="process at most N subjects/class")
    ap.add_argument("--summary", default=config.STEP2_SUMMARY)
    args = ap.parse_args(argv)

    if not os.path.exists(args.manifest):
        sys.exit(f"manifest not found ({args.manifest}) - run step1_collect_data.py")
    rows = read_manifest(args.manifest)
    if args.max:
        by_class = {}
        for r in rows:
            by_class.setdefault(r["class_name"], []).append(r)
        rows = [r for c in by_class.values() for r in c[: args.max]]
    os.makedirs(args.out_dir, exist_ok=True)
    for c in config.LABELS:
        os.makedirs(os.path.join(args.out_dir, c), exist_ok=True)

    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(process_one)(r["path"], r["class_name"], args.out_dir)
        for r in rows
    )
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]
    methods = {}
    for r in ok:
        methods[r["method"]] = methods.get(r["method"], 0) + 1
    summary = {
        "processed": len(ok), "failed": len(failed), "methods": methods,
        "failures": [{"path": r["path"], "error": r.get("error", "?")} for r in failed],
        "out_dir": os.path.abspath(args.out_dir), "seconds": round(time.time() - t0, 1),
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
