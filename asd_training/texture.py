import math

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.stats import kurtosis, skew
from skimage.feature import graycoprops, local_binary_pattern


GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")


def masked_glcm_counts(gray, mask, levels=8, distance=1, angle=0.0):
    quantized = np.minimum(np.floor(np.clip(gray, 0, 1) * levels), levels - 1).astype(np.int64)
    dr, dc = int(round(distance * math.sin(angle))), int(round(distance * math.cos(angle)))
    height, width = gray.shape
    y0, y1 = max(0, -dr), min(height, height - dr)
    x0, x1 = max(0, -dc), min(width, width - dc)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("No GLCM pixel pairs at this offset")
    valid = mask[y0:y1, x0:x1] & mask[y0 + dr:y1 + dr, x0 + dc:x1 + dc]
    if not valid.any():
        raise ValueError("No masked GLCM pixel pairs")
    source = quantized[y0:y1, x0:x1][valid]
    target = quantized[y0 + dr:y1 + dr, x0 + dc:x1 + dc][valid]
    counts = np.bincount(source * levels + target, minlength=levels * levels).reshape(levels, levels)
    counts = (counts + counts.T).astype(np.float64)
    return counts / counts.sum()


def slice_glcm(gray, mask, settings):
    """20-D: 4 angles x 5 properties at distance 1, 8 gray levels."""
    values, names = [], []
    for angle in settings.glcm_angles:
        matrix = masked_glcm_counts(gray, mask, settings.glcm_levels, 1, angle)
        expanded = matrix[:, :, None, None]
        for prop in settings.glcm_properties:
            values.append(float(graycoprops(expanded, prop)[0, 0]))
            names.append(f"glcm_a{int(round(math.degrees(angle))):03d}_{prop}")
    return values, names


def _lbp_codes(gray, mask, points, radius):
    uint8 = np.rint(np.clip(gray, 0, 1) * 255).astype(np.uint8)
    safe = binary_erosion(mask, structure=np.ones((2 * int(np.ceil(radius)) + 3,) * 2))
    if safe.sum() < 32:
        raise ValueError("Insufficient complete LBP sampling neighborhoods")
    return local_binary_pattern(uint8, points, radius, method="uniform")[safe].astype(np.float64)


def slice_lbp(gray, mask, settings):
    """16-D: 10-bin uniform histogram (P=8,R=1) + 6 moments over (P=16,R=2) codes."""
    values, names = [], []
    points, radius = settings.lbp_points[0], settings.lbp_radii[0]
    codes = _lbp_codes(gray, mask, points, radius)
    histogram = np.bincount(codes.astype(int), minlength=settings.lbp_bins).astype(np.float64)
    histogram /= histogram.sum()
    values.extend(histogram.tolist())
    names.extend(f"lbp_hist_bin{i}" for i in range(settings.lbp_bins))

    points2, radius2 = settings.lbp_points[1], settings.lbp_radii[1]
    codes2 = _lbp_codes(gray, mask, points2, radius2)
    moments = np.asarray([
        float(codes2.mean()), float(codes2.std()), float(skew(codes2)),
        float(kurtosis(codes2)), float(np.mean(codes2 ** 2)) / float(points2 ** 2),
        float(-(histogram[histogram > 0] * np.log2(histogram[histogram > 0])).sum()),
    ], dtype=np.float32)
    stat_names = ("mean", "std", "skew", "kurtosis", "energy", "entropy")
    values.extend(moments.tolist())
    names.extend(f"lbp_moment_{name}" for name in stat_names)
    return values, names


def slice_texture(gray, mask, settings):
    glcm_values, glcm_names = slice_glcm(gray, mask, settings)
    lbp_values, lbp_names = slice_lbp(gray, mask, settings)
    values = np.asarray(glcm_values + lbp_values, dtype=np.float32)
    names = glcm_names + lbp_names
    if len(values) != 36 or not np.isfinite(values).all():
        raise ValueError("Invalid texture slice feature schema")
    return values, names


def extract_texture(prepared, settings):
    values, names = [], []
    for view in settings.views:
        slices = [slice_texture(gray, mask, settings) for gray, mask in prepared["views"][view]]
        stack = np.stack([item[0] for item in slices])
        for statistic, vector in (("mean", stack.mean(0)), ("std", stack.std(0))):
            values.extend(vector.tolist())
            names.extend(f"texture__{view}__{statistic}__{name}" for name in slices[0][1])
    vector = np.asarray(values, dtype=np.float32)
    expected = 72 * len(settings.views)
    if vector.shape != (expected,) or not np.isfinite(vector).all():
        raise ValueError("Invalid patient texture vector")
    return vector, names
