import csv
import hashlib

import nibabel as nib
import numpy as np

from asd_training.artifacts import sha256_file


def brain_volume(seed, shape=(40, 40, 40), radius=14.0):
    rng = np.random.default_rng(seed)
    centre = [(size - 1) / 2 for size in shape]
    grid = np.indices(shape).astype(float)
    inside = sum(((axis - c) / radius) ** 2 for axis, c in zip(grid, centre)) <= 1
    data = np.zeros(shape, dtype=np.float32)
    data[inside] = 100 + 60 * rng.random(int(inside.sum())) + 2 * grid[0][inside]
    return data


def write_nifti(path, data, affine=None, zooms=None, description=None):
    affine = np.diag([-1.0, 1.0, 1.0, 1.0]) if affine is None else affine
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    if zooms is not None:
        image.header.set_zooms(zooms)
    if description is not None:
        image.header["descrip"] = description
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))
    return path


def write_phenotype(path, entries):
    fields = ["SUB_ID", "SITE_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX", "FIQ"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            writer.writerow({**{"SITE_ID": "S1", "AGE_AT_SCAN": 12.5, "SEX": 1, "FIQ": 100}, **entry})
    return path


def disc_slice(seed, size=40, radius=15):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size))
    mask = (yy - (size - 1) / 2) ** 2 + (xx - (size - 1) / 2) ** 2 <= radius ** 2
    gray = np.where(mask, rng.random((size, size)), 0).astype(np.float32)
    return gray, mask


def synthetic_prepared(seed, per_view=3, size=40, views=("sagittal",)):
    return {
        "views": {
            view: [disc_slice(seed * 1000 + 10 * v + k, size) for k in range(per_view)]
            for v, view in enumerate(views)
        },
        "qc": {"voxel_sha256": hashlib.sha256(str(seed).encode()).hexdigest()},
    }


class StubEncoder:
    def __init__(self, name, weights_path, dimension=4):
        self.name = name
        self.dimension = dimension
        self.weights_path = weights_path
        self.calls = 0
        self.closed = False

    def metadata(self):
        return {
            "backbone": self.name, "dimension": self.dimension, "image_size": 224,
            "weight_sha256": sha256_file(self.weights_path), "weights_path": str(self.weights_path),
            "precision": "float32", "frozen": True, "pooling": "per-view-slice-mean-std",
            "transform": "stub",
        }

    def subject_features(self, prepared):
        self.calls += 1
        pooled = []
        for view, pairs in prepared["views"].items():
            stack = np.stack([np.resize(gray[mask][:100], 2 * self.dimension)
                              for gray, mask in pairs])
            pooled.extend((stack.mean(0), stack.std(0)))
        return np.concatenate(pooled).astype(np.float32)

    def close(self):
        self.closed = True
