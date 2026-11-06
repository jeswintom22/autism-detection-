import math

import numpy as np
import pytest
from skimage.feature import graycoprops

from asd_training import texture
from asd_training.config import Settings
from synthetic import disc_slice, synthetic_prepared


@pytest.fixture
def settings():
    return Settings()


def full(shape):
    return np.ones(shape, dtype=bool)


def test_checkerboard_known_answer():
    gray = np.array([[0, 1], [1, 0]], dtype=np.float32)
    P = texture.masked_glcm_counts(gray, full((2, 2)), levels=2, distance=1, angle=0)
    assert np.allclose(P, [[0, 0.5], [0.5, 0]])
    expanded = P[:, :, None, None]
    assert graycoprops(expanded, "contrast")[0, 0] == pytest.approx(1)
    assert graycoprops(expanded, "dissimilarity")[0, 0] == pytest.approx(1)
    assert graycoprops(expanded, "homogeneity")[0, 0] == pytest.approx(0.5)
    assert graycoprops(expanded, "energy")[0, 0] == pytest.approx(math.sqrt(0.5))
    assert graycoprops(expanded, "correlation")[0, 0] == pytest.approx(-1)


def test_masked_counts_hand_computed():
    gray = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.float32)
    mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=bool)
    P = texture.masked_glcm_counts(gray, mask, levels=2, distance=1, angle=0)
    assert np.allclose(P, [[1 / 3, 1 / 3], [1 / 3, 0]])


def test_vertical_offset_and_symmetry_normalisation():
    gray = np.array([[0, 1], [1, 1], [0, 0]], dtype=np.float32)
    P = texture.masked_glcm_counts(gray, full((3, 2)), levels=2, distance=1, angle=math.pi / 2)
    assert np.allclose(P, P.T)
    assert P.sum() == pytest.approx(1)
    assert np.allclose(P * 8, [[0, 3], [3, 2]])


def test_masked_pairs_exclude_background_and_padding_invariance():
    gray, mask = disc_slice(1)
    padded_gray = np.pad(gray, 7)
    padded_mask = np.pad(mask, 7)
    for distance, angle in ((1, 0.0), (2, math.pi / 4), (4, math.pi / 2)):
        a = texture.masked_glcm_counts(gray, mask, 32, distance, angle)
        b = texture.masked_glcm_counts(padded_gray, padded_mask, 32, distance, angle)
        assert np.allclose(a, b)
    altered = np.where(mask, gray, 0.9)
    assert np.allclose(texture.masked_glcm_counts(gray, mask), texture.masked_glcm_counts(altered, mask))


def test_valid_zero_intensity_is_not_discarded():
    gray = np.zeros((4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    P = texture.masked_glcm_counts(gray, mask, levels=32, distance=1, angle=0)
    assert P[0, 0] == pytest.approx(1)


def test_top_intensity_falls_in_last_level():
    gray = np.ones((2, 2), dtype=np.float32)
    P = texture.masked_glcm_counts(gray, full((2, 2)), levels=32, distance=1, angle=0)
    assert P[31, 31] == pytest.approx(1)


def test_no_valid_pairs_is_an_error():
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = mask[2, 2] = True
    with pytest.raises(ValueError):
        texture.masked_glcm_counts(np.zeros((3, 3), dtype=np.float32), mask)


def test_glcm_slice_has_20_disaster_schema_features(settings):
    gray, mask = disc_slice(3)
    values, names = texture.slice_glcm(gray, mask, settings)
    assert len(values) == len(names) == 20
    assert all(n.startswith("glcm_a") for n in names)
    for angle in ("000", "045", "090", "135"):
        segment = [n for n in names if n.startswith(f"glcm_a{angle}_")]
        assert len(segment) == 5
    assert np.isfinite(values).all()


def test_lbp_slice_has_16_disaster_schema_features(settings):
    gray, mask = disc_slice(3)
    values, names = texture.slice_lbp(gray, mask, settings)
    assert len(values) == len(names) == 16
    assert sum(n.startswith("lbp_hist_bin") for n in names) == 10
    assert sum(n.startswith("lbp_moment_") for n in names) == 6
    hist = vector_hist(values, names)
    assert hist.sum() == pytest.approx(1, abs=1e-6)
    assert np.isfinite(values).all()


def vector_hist(values, names):
    return np.asarray([v for v, n in zip(values, names) if n.startswith("lbp_hist_bin")])


def test_slice_texture_dimensions_names_and_determinism(settings):
    gray, mask = disc_slice(3)
    vector, names = texture.slice_texture(gray, mask, settings)
    assert vector.shape == (36,) and len(names) == 36 and len(set(names)) == 36
    assert np.isfinite(vector).all()
    assert sum(n.startswith("glcm") for n in names) == 20
    assert sum(n.startswith("lbp") for n in names) == 16
    again, _ = texture.slice_texture(gray, mask, settings)
    assert np.array_equal(vector, again)


def test_slice_texture_ignores_everything_outside_the_eroded_mask(settings):
    gray, mask = disc_slice(4)
    reference, _ = texture.slice_texture(gray, mask, settings)
    altered = np.where(mask, gray, 0.77).astype(np.float32)
    assert np.array_equal(reference, texture.slice_texture(altered, mask, settings)[0])
    padded, _ = texture.slice_texture(np.pad(gray, 9), np.pad(mask, 9), settings)
    assert np.allclose(reference, padded)


def test_tiny_mask_fails_lbp_quality_gate(settings):
    gray = np.zeros((40, 40), dtype=np.float32)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:16, 10:16] = True
    gray[mask] = 0.5
    with pytest.raises(ValueError):
        texture.slice_texture(gray, mask, settings)


def test_patient_texture_has_72_ordered_sagittal_features(settings):
    prepared = synthetic_prepared(1, per_view=3)
    vector, names = texture.extract_texture(prepared, settings)
    assert vector.shape == (72,) and len(names) == 72 and len(set(names)) == 72
    assert np.isfinite(vector).all()
    assert all(n.startswith("texture__sagittal__") for n in names)
    assert [n.split("__")[2] for n in names] == ["mean"] * 36 + ["std"] * 36
    again, again_names = texture.extract_texture(synthetic_prepared(1, per_view=3), settings)
    assert np.array_equal(vector, again) and names == again_names


def test_pooling_is_mean_and_population_std(settings):
    prepared = synthetic_prepared(2, per_view=3)
    vector, _ = texture.extract_texture(prepared, settings)
    slices = np.stack([texture.slice_texture(g, m, settings)[0]
                       for g, m in prepared["views"]["sagittal"]])
    assert np.allclose(vector[:36], slices.mean(0), atol=1e-6)
    assert np.allclose(vector[36:], slices.std(0), atol=1e-6)
