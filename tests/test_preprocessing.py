import numpy as np
import pytest

from asd_training import preprocessing
from asd_training.config import Settings
from synthetic import brain_volume, write_nifti

@pytest.fixture
def settings():
    return Settings()


def ras_volume():
    data = brain_volume(5)
    data[20:26, 12:30, 16:24] += 40
    return data


def store(tmp_path, name, ras, flips=(False, False, False), perm=(0, 1, 2)):
    shape = ras.shape
    data = ras
    affine = np.eye(4)
    for axis, flip in enumerate(flips):
        if flip:
            data = np.flip(data, axis)
            affine[axis, axis] = -1
            affine[axis, 3] = shape[axis] - 1
    if perm != (0, 1, 2):
        data = np.transpose(data, perm)
        permuted = np.zeros((4, 4))
        permuted[3, 3] = 1
        for new_axis, old_axis in enumerate(perm):
            permuted[:3, new_axis] = affine[:3, old_axis]
        permuted[:3, 3] = affine[:3, 3]
        affine = permuted
    return write_nifti(tmp_path / f"{name}.nii.gz", data, affine=affine)


def prepared_views(path, settings):
    return preprocessing.prepare_subject(path, settings)["views"]


@pytest.mark.parametrize("flips,perm", [
    ((True, False, False), (0, 1, 2)),
    ((False, True, True), (0, 1, 2)),
    ((False, False, False), (1, 0, 2)),
    ((True, False, False), (2, 0, 1)),
])
def test_orientations_map_to_a_common_ras_grid(tmp_path, settings, flips, perm):
    ras = ras_volume()
    reference = prepared_views(store(tmp_path, "ref", ras), settings)
    other = prepared_views(store(tmp_path, "other", ras, flips, perm), settings)
    for view in settings.views:
        assert len(reference[view]) == len(other[view]) == settings.n_slices
        for (g1, m1), (g2, m2) in zip(reference[view], other[view]):
            assert g1.shape == g2.shape and (m1 == m2).all()
            assert np.allclose(g1, g2, atol=1e-5)


def test_las_input_matches_ras(tmp_path, settings):
    ras = ras_volume()
    affine = np.array([[-1.0, 0, 0, 39], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    las = write_nifti(tmp_path / "las.nii.gz", np.flip(ras, 0), affine=affine)
    reference = prepared_views(store(tmp_path, "ref", ras), settings)
    other = prepared_views(las, settings)
    for view in settings.views:
        for (g1, _), (g2, _) in zip(reference[view], other[view]):
            assert np.allclose(g1, g2, atol=1e-5)


def test_anisotropic_input_resamples_to_one_millimetre_grid(tmp_path, settings):
    data = brain_volume(6, shape=(30, 30, 20), radius=10)
    affine = np.diag([2.0, 2.0, 1.0, 1.0])
    prepared = preprocessing.prepare_subject(write_nifti(tmp_path / "a.nii.gz", data, affine=affine), settings)
    qc = prepared["qc"]
    assert qc["resampled_shape"][0] >= 55 and qc["resampled_shape"][2] <= 25
    zooms = np.sqrt((np.asarray(qc["output_affine"])[:3, :3] ** 2).sum(0))
    assert np.allclose(zooms, 1.0)


def test_mask_is_boolean_aligned_and_background_is_exactly_zero(tmp_path, settings):
    views = prepared_views(store(tmp_path, "a", ras_volume()), settings)
    for view in settings.views:
        for gray, mask in views[view]:
            assert gray.shape == mask.shape and mask.dtype == bool
            assert (gray[~mask] == 0).all()
            assert gray.min() >= 0 and gray.max() <= 1
            assert gray[mask].max() > 0 and mask.sum() >= 256


def test_slice_positions_are_unique_ascending_and_recorded(tmp_path, settings):
    qc = preprocessing.prepare_subject(store(tmp_path, "a", ras_volume()), settings)["qc"]
    for view in settings.views:
        indices = qc["slice_indices"][view]
        assert len(indices) == settings.n_slices == len(set(indices))
        assert indices == sorted(indices)
    assert qc["preprocessing_version"] == settings.preprocessing_version


def test_preprocessing_is_deterministic_and_ignores_global_random_state(tmp_path, settings):
    path = store(tmp_path, "a", ras_volume())
    np.random.seed(1)
    first = preprocessing.prepare_subject(path, settings)
    np.random.seed(2)
    second = preprocessing.prepare_subject(path, settings)
    for view in settings.views:
        for (g1, m1), (g2, m2) in zip(first["views"][view], second["views"][view]):
            assert (g1 == g2).all() and (m1 == m2).all()
    assert first["qc"] == second["qc"]


def test_voxel_hash_is_opt_in_and_orientation_stable(tmp_path, settings):
    ras = ras_volume()
    plain = preprocessing.prepare_subject(store(tmp_path, "a", ras), settings)
    assert "voxel_sha256" not in plain["qc"]
    one = preprocessing.prepare_subject(store(tmp_path, "b", ras), settings, hash_voxels=True)
    two = preprocessing.prepare_subject(store(tmp_path, "c", ras, (True, False, False)), settings, hash_voxels=True)
    assert one["qc"]["voxel_sha256"] == two["qc"]["voxel_sha256"]


def test_thin_constant_empty_and_nonfinite_volumes_fail(tmp_path, settings):
    with pytest.raises(ValueError):
        preprocessing.prepare_subject(write_nifti(tmp_path / "thin.nii.gz", np.ones((40, 40, 4))), settings)
    with pytest.raises(ValueError):
        preprocessing.prepare_subject(write_nifti(tmp_path / "empty.nii.gz", np.zeros((40, 40, 40))), settings)
    constant = (brain_volume(1) > 0).astype(np.float32)
    with pytest.raises(ValueError):
        preprocessing.prepare_subject(write_nifti(tmp_path / "const.nii.gz", constant), settings)
    bad = brain_volume(1)
    bad[20, 20, 20] = np.inf
    with pytest.raises(ValueError):
        preprocessing.prepare_subject(write_nifti(tmp_path / "inf.nii.gz", bad), settings)
    small = brain_volume(2, shape=(40, 40, 40), radius=5)
    with pytest.raises(ValueError):
        preprocessing.prepare_subject(write_nifti(tmp_path / "small.nii.gz", small), settings)


def test_cnn_image_preserves_aspect_ratio_with_zero_letterbox():
    gray = np.ones((20, 40), dtype=np.float32)
    image = preprocessing.cnn_image(gray, 64)
    assert image.shape == (3, 64, 64) and image.dtype == np.float32
    assert (image[0] == image[1]).all() and (image[1] == image[2]).all()
    rows = np.flatnonzero(image[0].any(axis=1))
    cols = np.flatnonzero(image[0].any(axis=0))
    assert rows.max() - rows.min() + 1 == 32 and cols.max() - cols.min() + 1 == 64
    assert (image[0][:rows.min()] == 0).all() and (image[0][rows.max() + 1:] == 0).all()
    assert abs(rows.min() - (64 - 1 - rows.max())) <= 1


def test_cnn_image_square_slice_fills_canvas_without_cropping():
    gray = np.full((30, 30), 0.5, dtype=np.float32)
    image = preprocessing.cnn_image(gray, 224)
    assert image.shape == (3, 224, 224)
    assert np.allclose(image[0], 0.5)
