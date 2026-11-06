import numpy as np
import cv2
from scipy import ndimage


LMS_MATRIX = np.array([
    [0.300, 0.622, 0.078],
    [0.230, 0.692, 0.078],
    [0.243, 0.205, 0.553],
], dtype=np.float32)
PYRAMID_LEVELS = (2, 3, 4, 5)


def _normalize01(image):
    low, high = float(image.min()), float(image.max())
    if not np.isfinite([low, high]).all() or high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - low) / (high - low)).astype(np.float32)


def saliency_map(gray):
    """LMS opponent-pyramid saliency for a single [0, 1] grayscale slice."""
    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("Saliency requires a non-empty 2-D slice")
    rgb = np.repeat(gray[..., None], 3, axis=2)
    lms = rgb @ LMS_MATRIX.T
    lms = np.clip(lms, 0.0, 1.0)
    long, medium, short = lms[..., 0], lms[..., 1], lms[..., 2]
    opponent = np.stack([
        long - medium,
        short - (long + medium) / 2.0,
        (long + medium + short) / 3.0,
    ], axis=2)
    response = np.zeros(gray.shape, dtype=np.float32)
    for channel in range(3):
        plane = opponent[..., channel]
        for level in PYRAMID_LEVELS:
            factor = 1 << level
            small_h = max(1, gray.shape[0] // factor)
            small_w = max(1, gray.shape[1] // factor)
            down = cv2.resize(plane, (small_w, small_h), interpolation=cv2.INTER_AREA)
            surround = cv2.resize(down, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
            response += np.abs(plane - surround)
    return _normalize01(response)


def saliency_stats(saliency):
    values = np.asarray(saliency, dtype=np.float64).ravel()
    total = float(values.sum())
    probs = values / total if total > 0 else values
    centered = values - values.mean()
    variance = float(np.mean(centered ** 2))
    skew = float(np.mean(centered ** 3) / variance ** 1.5) if variance > 0 else 0.0
    kurt = float(np.mean(centered ** 4) / variance ** 2 - 3.0) if variance > 0 else 0.0
    nonzero = probs[probs > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero))) if nonzero.size else 0.0
    return np.asarray([
        float(values.mean()), float(values.std()), skew, kurt,
        float(np.mean(values ** 2)), entropy,
    ], dtype=np.float32)


def enhance_slice(gray, alpha):
    """Return (enhanced_gray in [0, 1], saliency_map) for one slice."""
    saliency = saliency_map(gray)
    enhanced = np.clip(alpha * gray + (1.0 - alpha) * gray * saliency, 0.0, 1.0)
    return enhanced.astype(np.float32), saliency


def extract_saliency(prepared, settings):
    """Per-patient saliency block: enhanced CNN inputs + pooled stats."""
    values, names = [], []
    cnn_views = {}
    for view in settings.views:
        pairs = prepared["views"][view]
        enhanced_pairs, stats = [], []
        for gray, mask in pairs:
            enhanced, saliency = enhance_slice(gray, settings.saliency_alpha)
            enhanced_pairs.append((np.ascontiguousarray(enhanced), mask))
            stats.append(saliency_stats(saliency))
        cnn_views[view] = enhanced_pairs
        stack = np.stack(stats)
        stat_names = ("mean", "std", "skew", "kurtosis", "energy", "entropy")
        values.extend(stack.mean(0).tolist())
        names.extend(f"saliency__{view}__mean__{name}" for name in stat_names)
        values.extend(stack.std(0).tolist())
        names.extend(f"saliency__{view}__std__{name}" for name in stat_names)
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (12 * len(settings.views),) or not np.isfinite(vector).all():
        raise ValueError("Invalid patient saliency vector")
    prepared = dict(prepared)
    prepared["cnn_views"] = cnn_views
    return vector, names, prepared
