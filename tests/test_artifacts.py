import json
import os

import pytest

from asd_training import artifacts
from asd_training.config import Settings


@pytest.fixture
def settings(tmp_path):
    repo = tmp_path / "repo"
    (repo / "runs" / "hybrid").mkdir(parents=True)
    for name in ("saved_models_b5", "results", "slice_cache", "preprocessed_dataset", "control", "treatment"):
        (repo / name).mkdir()
    return Settings(
        repo_root=repo,
        control_dir=repo / "control",
        treatment_dir=repo / "treatment",
    )


def test_stable_hash_is_order_independent_and_sensitive():
    a = {"b": [1, 2], "a": {"y": 1.5, "x": None}}
    b = {"a": {"x": None, "y": 1.5}, "b": [1, 2]}
    assert artifacts.stable_hash(a) == artifacts.stable_hash(b)
    assert artifacts.stable_hash(a) != artifacts.stable_hash({**a, "b": [2, 1]})
    assert len(artifacts.stable_hash(a)) == 64


def test_stable_hash_rejects_nan():
    with pytest.raises(ValueError):
        artifacts.stable_hash({"v": float("nan")})


def test_sha256_file_streams_known_digest(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert artifacts.sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_atomic_json_roundtrip_and_no_temp_leftovers(tmp_path):
    path = tmp_path / "out" / "a.json"
    artifacts.atomic_json(path, {"k": [1, 2], "n": None})
    assert artifacts.read_json(path) == {"k": [1, 2], "n": None}
    assert [p.name for p in path.parent.iterdir()] == ["a.json"]


def test_atomic_json_failure_keeps_original(tmp_path):
    path = tmp_path / "a.json"
    artifacts.atomic_json(path, {"ok": 1})
    with pytest.raises(ValueError):
        artifacts.atomic_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"ok": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_atomic_json_refuses_non_json_and_directories(tmp_path):
    with pytest.raises(ValueError):
        artifacts.atomic_json(tmp_path / "a.nii.gz", {})
    (tmp_path / "d.json").mkdir()
    with pytest.raises(IsADirectoryError):
        artifacts.atomic_json(tmp_path / "d.json", {})


def test_atomic_json_refuses_legacy_repo_paths():
    root = artifacts.REPO_ROOT
    with pytest.raises(PermissionError):
        artifacts.atomic_json(root / "results_fused.json", {})
    with pytest.raises(PermissionError):
        artifacts.atomic_json(root / "results_b5" / "x.json", {})


def test_atomic_csv_writes_header_and_flattens(tmp_path):
    path = tmp_path / "r.csv"
    artifacts.atomic_csv(path, [{"a": 1, "g": {"z": 1}}, {"a": None, "b": "x"}])
    lines = path.read_text().splitlines()
    assert lines[0] == "a,g,b"
    assert len(lines) == 3
    assert lines[2] == ",,x"


def test_atomic_csv_explicit_fieldnames_and_empty(tmp_path):
    path = tmp_path / "e.csv"
    artifacts.atomic_csv(path, [], fieldnames=["a", "b"])
    assert path.read_text().splitlines() == ["a,b"]


def test_ensure_safe_output_allows_production_and_temp(settings, tmp_path):
    prod = settings.repo_root / "runs" / "hybrid" / "r1"
    assert artifacts.ensure_safe_output(prod, settings) == prod
    assert artifacts.ensure_safe_output(tmp_path / "smoke", settings) == tmp_path / "smoke"


@pytest.mark.parametrize("rel", [
    "control", "treatment/sub", "saved_models_b5", "saved_models_b5/x", "results", "results/x/y",
    "slice_cache", "preprocessed_dataset/a", "runs", "other",
])
def test_ensure_safe_output_rejects_protected_and_outside(settings, rel):
    with pytest.raises(PermissionError):
        artifacts.ensure_safe_output(settings.repo_root / rel, settings)


def test_ensure_safe_output_rejects_non_temp_outside_repo(settings):
    outside = os.path.abspath(os.sep + "asd_outside_output")
    with pytest.raises(PermissionError):
        artifacts.ensure_safe_output(outside, settings)


def test_atomic_writers_refuse_arbitrary_locations(settings):
    outside = os.path.abspath(os.sep + "asd_outside_output")
    with pytest.raises(PermissionError):
        artifacts.atomic_json(os.path.join(outside, "a.json"), {})
    with pytest.raises(PermissionError):
        artifacts.atomic_csv(os.path.join(outside, "a.csv"), [])
    with pytest.raises(PermissionError):
        artifacts.atomic_text(os.path.join(outside, "a.md"), "x")
    with pytest.raises(PermissionError):
        artifacts.atomic_json(artifacts.REPO_ROOT / "runs" / "other" / "a.json", {})
    with pytest.raises(PermissionError):
        artifacts.atomic_json(artifacts.REPO_ROOT / "config_new.json", {})


def test_allowed_roots_cannot_be_run_dirs(settings):
    import tempfile
    with pytest.raises(PermissionError):
        artifacts.ensure_safe_output(settings.repo_root / "runs" / "hybrid", settings)
    with pytest.raises(PermissionError):
        artifacts.ensure_safe_output(tempfile.gettempdir(), settings)
    with pytest.raises(PermissionError):
        artifacts.ensure_allowed_output(artifacts.REPO_ROOT / "runs" / "hybrid")


def test_atomic_writers_do_not_touch_raw_inputs(settings):
    scan = settings.control_dir / "1_skull_stripped.nii.gz"
    scan.write_bytes(b"raw")
    stamp = (scan.read_bytes(), scan.stat().st_mtime_ns)
    with pytest.raises(PermissionError):
        artifacts.ensure_safe_output(scan.parent / "run", settings)
    assert (scan.read_bytes(), scan.stat().st_mtime_ns) == stamp
