import shutil

import numpy as np
import pytest

from asd_training import dataset
from asd_training.config import Settings
from synthetic import brain_volume, write_nifti


def make_settings(tmp_path):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    return Settings(
        repo_root=root, control_dir=tmp_path / "control", treatment_dir=tmp_path / "treatment",
    )


def scan(tmp_path, cls, sid, seed, **kwargs):
    return write_nifti(tmp_path / cls / f"{sid}_skull_stripped.nii.gz", brain_volume(seed), **kwargs)


@pytest.fixture
def cohort(tmp_path):
    for sid in (1, 2):
        scan(tmp_path, "control", sid, sid)
    for sid in (3, 4):
        scan(tmp_path, "treatment", sid, 10 + sid)
    return tmp_path, make_settings(tmp_path)


def errors_text(report):
    return " | ".join(report["blocking_errors"])


def snapshot(root):
    return {p: (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob("*") if p.is_file()}


def test_normalize_subject_id_removes_leading_zeros_and_rejects_unknown():
    assert dataset.normalize_subject_id("0050") == "50"
    assert dataset.normalize_subject_id(" 7 ") == "7"
    for bad in ("", "abc", "12a", "-3", "1.5"):
        with pytest.raises(ValueError):
            dataset.normalize_subject_id(bad)


def test_clean_cohort_passes_and_rows_have_contract_fields(cohort):
    tmp_path, settings = cohort
    rows, report = dataset.audit_dataset(settings)
    assert report["blocking_errors"] == [] and report["failures"] == []
    assert [r["subject_id"] for r in rows] == ["1", "2", "3", "4"]
    assert [r["label"] for r in rows] == [0, 0, 1, 1]
    for key in ("class_name", "raw_path", "source_sha256", "voxel_sha256",
                "source_size", "source_mtime_ns", "geometry", "person_id"):
        assert key in rows[0]
    assert "site_id" not in rows[0] and "age" not in rows[0]
    assert len({r["source_sha256"] for r in rows}) == 4 and len({r["voxel_sha256"] for r in rows}) == 4
    assert report["patients"] == 4 and report["unique_persons"] == 4
    dataset.verify_sources(rows)


def test_audit_does_not_mutate_or_write(cohort):
    tmp_path, settings = cohort
    before = snapshot(tmp_path)
    dataset.audit_dataset(settings)
    assert snapshot(tmp_path) == before
    assert list(settings.repo_root.iterdir()) == []


def test_unexpected_filename_is_blocking(cohort):
    tmp_path, settings = cohort
    shutil.copy(tmp_path / "control" / "1_skull_stripped.nii.gz", tmp_path / "control" / "weird_scan.nii.gz")
    _, report = dataset.audit_dataset(settings)
    assert "Unexpected raw filename" in errors_text(report)


def test_duplicate_id_across_folders_is_blocking_and_not_dropped_silently(cohort):
    tmp_path, settings = cohort
    scan(tmp_path, "treatment", "0001", 77)
    rows, report = dataset.audit_dataset(settings)
    assert "Duplicate patient ID" in errors_text(report)
    assert "1" not in {r["subject_id"] for r in rows}


def test_leading_zero_does_not_create_second_identity(cohort):
    tmp_path, settings = cohort
    scan(tmp_path, "control", "00002", 88)
    _, report = dataset.audit_dataset(settings)
    assert "Duplicate patient ID" in errors_text(report)


def test_unreadable_volume_is_reported(cohort):
    tmp_path, settings = cohort
    (tmp_path / "control" / "1_skull_stripped.nii.gz").write_bytes(b"not a nifti")
    rows, report = dataset.audit_dataset(settings)
    assert [f["subject_id"] for f in report["failures"]] == ["1"]
    assert report["blocking_errors"]


def test_nonfinite_empty_and_constant_volumes_are_reported(cohort):
    tmp_path, settings = cohort
    bad = brain_volume(1)
    bad[20, 20, 20] = np.nan
    write_nifti(tmp_path / "control" / "1_skull_stripped.nii.gz", bad)
    write_nifti(tmp_path / "control" / "2_skull_stripped.nii.gz", np.zeros((40, 40, 40)))
    constant = (brain_volume(13) > 0).astype(np.float32)
    write_nifti(tmp_path / "treatment" / "3_skull_stripped.nii.gz", constant)
    rows, report = dataset.audit_dataset(settings)
    failed = {f["subject_id"]: f["error"] for f in report["failures"]}
    assert set(failed) == {"1", "2", "3"}
    assert "nonfinite" in failed["1"].lower()
    assert "Empty" in failed["2"]
    assert "Constant" in failed["3"]
    assert [r["subject_id"] for r in rows] == ["4"]


def test_exact_file_duplicate_under_different_labels_is_blocking(cohort):
    tmp_path, settings = cohort
    shutil.copy(tmp_path / "control" / "1_skull_stripped.nii.gz", tmp_path / "treatment" / "3_skull_stripped.nii.gz")
    _, report = dataset.audit_dataset(settings)
    text = errors_text(report)
    assert "conflicting identity" in text and "Duplicate source_sha256" in text


def test_voxel_duplicate_with_different_file_bytes_is_blocking(cohort):
    tmp_path, settings = cohort
    write_nifti(tmp_path / "treatment" / "3_skull_stripped.nii.gz", brain_volume(1), description=b"other header")
    _, report = dataset.audit_dataset(settings)
    text = errors_text(report)
    assert "Duplicate voxel_sha256" in text and "Duplicate source_sha256" not in text


def test_missing_class_folder_is_blocking(cohort):
    tmp_path, settings = cohort
    shutil.rmtree(tmp_path / "treatment")
    _, report = dataset.audit_dataset(settings)
    assert "Missing treatment source directory" in errors_text(report)


def test_verify_sources_detects_changed_scan(cohort):
    tmp_path, settings = cohort
    rows, _ = dataset.audit_dataset(settings)
    path = tmp_path / "control" / "1_skull_stripped.nii.gz"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError):
        dataset.verify_sources(rows)


def test_same_label_duplicate_is_grouped_as_one_person(cohort):
    tmp_path, settings = cohort
    shutil.copy(tmp_path / "treatment" / "3_skull_stripped.nii.gz",
                tmp_path / "treatment" / "5_skull_stripped.nii.gz")
    rows, report = dataset.audit_dataset(settings)
    assert report["blocking_errors"] == []
    assert report["patients"] == 5 and report["unique_persons"] == 4
    person = {r["subject_id"]: r["person_id"] for r in rows}
    assert person["3"] == person["5"] == "3" and person["4"] == "4"
    assert report["duplicate_groups"] == [{"subject_ids": ["3", "5"], "person_id": "3"}]
    assert report["warnings"]


def test_conflicting_label_duplicate_is_blocking(cohort):
    tmp_path, settings = cohort
    shutil.copy(tmp_path / "control" / "1_skull_stripped.nii.gz",
                tmp_path / "treatment" / "5_skull_stripped.nii.gz")
    _, report = dataset.audit_dataset(settings)
    text = errors_text(report)
    assert "conflicting identity" in text and "1, 5" in text
