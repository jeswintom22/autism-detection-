"""Step 1 - collect and validate the NIfTI dataset.

Scans CONTROL_DIR and TREATMENT_DIR (config.py), verifies every .nii/.nii.gz
file loads, and writes:
    manifest.csv        one row per subject: id, label, path, shape, stats
    step1_summary.json  per-class counts + failures

No data is copied - the dataset is already organised as <class>/<scan>.nii.gz.
"""
import argparse
import csv
import json
import os
import re
import sys

import nibabel as nib
import numpy as np

import config

FIELDNAMES = [
    "subject_id", "label", "class_name", "path",
    "shape", "n_voxels", "nonzero_frac", "status", "error",
]


def find_nii(folder):
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith((".nii", ".nii.gz")):
                out.append(os.path.join(root, f))
    return sorted(out)


def subject_id(path):
    m = re.match(r"(\d+)", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def validate(path):
    """Header + middle-slice read: catches corrupt headers and bodies."""
    img = nib.load(path)
    shape = tuple(int(s) for s in img.shape)
    if len(shape) < 3 or min(shape[:3]) < 8:
        raise ValueError(f"bad shape {shape}")
    mid = np.asarray(img.dataobj[:, :, shape[2] // 2])
    nz = float((mid > 0).mean())
    return shape, nz


def collect(control_dir, treatment_dir, manifest_path):
    rows, failures = [], []
    for class_name, label in config.LABELS.items():
        folder = {"control": control_dir, "treatment": treatment_dir}[class_name]
        files = find_nii(folder)
        print(f"{class_name}: {len(files)} files")
        for i, path in enumerate(files, 1):
            row = {
                "subject_id": subject_id(path),
                "label": label,
                "class_name": class_name,
                "path": os.path.abspath(path),
                "shape": "",
                "n_voxels": "",
                "nonzero_frac": "",
                "status": "ok",
                "error": "",
            }
            try:
                shape, nz = validate(path)
                row["shape"] = "x".join(map(str, shape))
                row["n_voxels"] = int(np.prod(shape))
                row["nonzero_frac"] = f"{nz:.4f}"
            except Exception as e:  # noqa: BLE001 - record and continue
                row["status"] = "fail"
                row["error"] = str(e)
                failures.append(path)
            rows.append(row)
            if i % 100 == 0:
                print(f"  {class_name}: {i}/{len(files)}")
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    return rows, failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", default=config.CONTROL_DIR)
    ap.add_argument("--treatment", default=config.TREATMENT_DIR)
    ap.add_argument("--manifest", default=config.MANIFEST_PATH)
    ap.add_argument("--summary", default=config.STEP1_SUMMARY)
    args = ap.parse_args(argv)

    for d in (args.control, args.treatment):
        if not os.path.isdir(d):
            sys.exit(f"missing directory: {d}")

    rows, failures = collect(args.control, args.treatment, args.manifest)
    ok = [r for r in rows if r["status"] == "ok"]
    summary = {
        "control": {"total": sum(r["class_name"] == "control" for r in rows),
                    "ok": sum(r["class_name"] == "control" for r in ok)},
        "treatment": {"total": sum(r["class_name"] == "treatment" for r in rows),
                      "ok": sum(r["class_name"] == "treatment" for r in ok)},
        "failures": failures,
        "manifest": os.path.abspath(args.manifest),
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"manifest -> {args.manifest}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
