import hashlib
import os
import shutil
import subprocess
import tempfile

import cv2
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output
from scipy import ndimage
from scipy.ndimage import binary_erosion

from .config import ALL_VIEWS


AXES = {"axial": 2, "coronal": 1, "sagittal": 0}
_NEIGHBOUR4 = np.array([[0, 1, 0], [1, 0, 1], [1, 0, 1]], np.float32) / 4.0


def nimble_filter_slice(gray, window, g, canny_low, canny_high):
    """Adaptive nimble filter for one 2-D slice; input/output in [0, 1]."""
    f = cv2.normalize(gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    m = cv2.filter2D(f, cv2.CV_32F, _NEIGHBOUR4, borderType=cv2.BORDER_REPLICATE)
    edges = cv2.Canny(np.clip(f, 0, 255).astype(np.uint8), canny_low, canny_high)
    density = cv2.boxFilter((edges > 0).astype(np.float32), cv2.CV_32F, (window, window))
    mean = cv2.boxFilter(f, cv2.CV_32F, (window, window))
    sq = cv2.boxFilter(f * f, cv2.CV_32F, (window, window))
    noise = np.sqrt(np.clip(sq - mean * mean, 0, None)) / 255.0
    t = np.clip(density * (1.0 - noise), 0.0, 1.0)
    return (np.clip(g * f + (1.0 - g) * m + t * (f - m), 0, 255) / 255.0).astype(np.float32)


def nimble_filter_volume(volume, settings):
    filtered = np.empty_like(volume)
    for z in range(volume.shape[2]):
        filtered[:, :, z] = nimble_filter_slice(
            volume[:, :, z], settings.nimble_window, settings.nimble_g,
            settings.canny_low, settings.canny_high,
        )
    return filtered


def _is_stripped(volume, settings):
    nz_frac = float((volume > 0).mean())
    s = settings.strip_rim_slab
    rim = np.concatenate([
        volume[:s].ravel(), volume[-s:].ravel(),
        volume[:, :s].ravel(), volume[:, -s:].ravel(),
        volume[:, :, :s].ravel(), volume[:, :, -s:].ravel(),
    ])
    rim_frac = float((rim > 0).mean())
    return nz_frac <= settings.strip_max_nonzero and rim_frac <= settings.strip_max_rim


def _strip_morphology(volume):
    positive = volume[volume > 0]
    if positive.size == 0:
        return volume
    mask = volume > np.percentile(positive, 35)
    mask = ndimage.binary_opening(mask, iterations=2)
    labels, count = ndimage.label(mask)
    if count == 0:
        return volume
    biggest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
    mask = labels == biggest
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_erosion(mask, iterations=1)
    return np.where(mask, volume, 0).astype(np.float32)


def _strip_external(volume, affine, tool):
    """Run an external skull-stripper (FSL bet or hd-bet) on the resampled volume."""
    with tempfile.TemporaryDirectory(prefix="asd_strip_") as tmp:
        source = os.path.join(tmp, "in.nii.gz")
        output = os.path.join(tmp, "out.nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(volume, dtype=np.float32), affine), source)
        if tool == "hd-bet":
            command = ["hd-bet", "-i", source, "-o", output, "--disable_tta"]
        else:
            command = ["bet", source, output, "-R", "-f", "0.45"]
        subprocess.run(command, check=True, capture_output=True)
        stripped = nib.load(output).get_fdata(dtype=np.float32)
    if stripped.shape != volume.shape or not np.isfinite(stripped).all() or not (stripped > 0).any():
        raise ValueError(f"{tool} returned an invalid stripped volume")
    return np.asarray(stripped, dtype=np.float32)


def deskull_check(volume, settings, affine=None):
    """Return (stripped_volume, method); prefers FSL bet, then hd-bet, then morphology."""
    if _is_stripped(volume, settings):
        return volume, "passed"
    if affine is not None:
        for tool, method in (("bet", "fsl-bet"), ("hd-bet", "hd-bet")):
            if not shutil.which(tool):
                continue
            try:
                return _strip_external(volume, affine, tool), method
            except (OSError, subprocess.SubprocessError, ValueError):
                continue
    return _strip_morphology(volume), "morphology"


def prepare_subject(path, settings, hash_voxels=False):
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    affine = image.affine
    if data.ndim != 3 or min(data.shape) < 8:
        raise ValueError(f"Expected a nontrivial 3-D MRI, got {data.shape}")
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError("Nonfinite or singular MRI affine")
    if not np.isfinite(data).all():
        raise ValueError("MRI contains nonfinite voxels")
    if not all(np.isfinite(z) and z > 0 for z in image.header.get_zooms()[:3]):
        raise ValueError("Invalid voxel spacing")
    canonical = nib.as_closest_canonical(nib.Nifti1Image(data, affine))
    geometry = {
        "shape": list(image.shape), "zooms": [float(z) for z in image.header.get_zooms()[:3]],
        "orientation": "".join(nib.aff2axcodes(affine)), "affine": affine.tolist(),
        "canonical_orientation": "".join(nib.aff2axcodes(canonical.affine)),
    }
    if hash_voxels:
        values = np.ascontiguousarray(np.asarray(canonical.dataobj), dtype="<f4")
        digest = hashlib.sha256(str(values.shape).encode())
        digest.update(memoryview(values).cast("B"))
        geometry["voxel_sha256"] = digest.hexdigest()
        del values
    foreground = np.asarray(canonical.dataobj) > 0
    if not foreground.any():
        raise ValueError("Empty skull-stripped foreground")
    positive = np.asarray(canonical.dataobj)[foreground]
    if np.ptp(positive) <= 0:
        raise ValueError("Constant MRI foreground")
    mask_image = nib.Nifti1Image(foreground.astype(np.uint8), canonical.affine)
    resampled = resample_to_output(
        canonical, voxel_sizes=(settings.spacing,) * 3, order=1, mode="constant", cval=0,
    )
    mask = np.asarray(resample_from_to(
        mask_image, (resampled.shape, resampled.affine), order=0, mode="constant", cval=0,
    ).dataobj) > 0
    volume = np.asarray(resampled.dataobj, dtype=np.float32)
    values = volume[mask]
    if values.size == 0:
        raise ValueError("No foreground after resampling")
    lo, hi = np.percentile(values, settings.percentiles)
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("Collapsed foreground normalization range")
    volume = np.where(mask, np.clip((volume - lo) / (hi - lo), 0, 1), 0).astype(np.float32)
    bounds = []
    margin = int(round(settings.margin_mm / settings.spacing))
    for axis in range(3):
        occupied = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        bounds.append((int(occupied[0]), int(occupied[-1])))
    crop = tuple(slice(max(0, a - margin), min(mask.shape[i], b + margin + 1))
                 for i, (a, b) in enumerate(bounds))
    volume, mask = volume[crop], mask[crop]
    volume = nimble_filter_volume(volume, settings)
    volume, deskull_method = deskull_check(volume, settings, resampled.affine)
    mask = volume > 0
    if not mask.any():
        raise ValueError("Empty foreground after skull-strip check")
    views, indices = {}, {}
    max_radius = max(settings.lbp_radii) if settings.lbp_radii else 1
    for name in settings.views:
        if name not in AXES:
            raise ValueError(f"Unknown view: {name}")
        axis = AXES[name]
        occupied = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        start, end = occupied[0], occupied[-1]
        selected = np.rint(np.linspace(
            start + 0.2 * (end - start), start + 0.8 * (end - start), settings.n_slices,
        )).astype(int)
        if len(np.unique(selected)) != settings.n_slices:
            raise ValueError(f"{name}: insufficient unique anatomical slices")
        pairs = []
        for index in selected:
            gray, foreground = np.take(volume, index, axis), np.take(mask, index, axis)
            if foreground.sum() < 256:
                raise ValueError(f"{name} slice {index}: insufficient brain pixels")
            safe = binary_erosion(foreground, structure=np.ones((2 * int(np.ceil(max_radius)) + 3,) * 2))
            if safe.sum() < 32:
                raise ValueError(f"{name} slice {index}: insufficient safe LBP centers")
            pairs.append((np.ascontiguousarray(gray), np.ascontiguousarray(foreground)))
        views[name] = pairs
        indices[name] = [int(i) for i in selected]
    geometry.update({
        "resampled_shape": list(resampled.shape), "output_affine": resampled.affine.tolist(),
        "percentile_bounds": [float(lo), float(hi)], "slice_indices": indices,
        "foreground_voxels": int(mask.sum()), "preprocessing_version": settings.preprocessing_version,
        "nimble": {"window": settings.nimble_window, "g": settings.nimble_g,
                   "canny": [settings.canny_low, settings.canny_high]},
        "deskull": deskull_method,
    })
    return {"views": views, "qc": geometry}


def cnn_image(gray, size):
    height, width = gray.shape
    scale = size / max(height, width)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = cv2.resize(gray, target, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.float32)
    top, left = (size - target[1]) // 2, (size - target[0]) // 2
    canvas[top:top + target[1], left:left + target[0]] = resized
    return np.repeat(canvas[None], 3, axis=0)
