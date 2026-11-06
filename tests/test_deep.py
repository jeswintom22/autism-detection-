import dataclasses
import time

import numpy as np
import pytest
import torch
from torchvision import models

from asd_training import deep, features
from asd_training.artifacts import sha256_file
from asd_training.config import Settings
from synthetic import StubEncoder, synthetic_prepared


@pytest.fixture
def settings():
    return Settings()


def bypass_encoder(settings):
    torch.manual_seed(0)
    model = models.efficientnet_b0(weights=None)
    model.classifier = torch.nn.Identity()
    model.requires_grad_(False).eval()
    encoder = object.__new__(deep.FrozenEncoder)
    encoder.name, encoder.settings, encoder.dimension, encoder.image_size = "efficientnet_b0", settings, 1280, 224
    encoder.device = torch.device("cpu")
    encoder.batch_size = 2
    encoder.model = model
    encoder.mean = torch.tensor((0.485, 0.456, 0.406))[None, :, None, None]
    encoder.std = torch.tensor((0.229, 0.224, 0.225))[None, :, None, None]
    encoder.weight_hash = "0" * 64
    encoder.weights_path = "offline"
    return encoder


def test_b0_adapter_embedding_dimension_pooling_and_frozen_state(settings):
    encoder = bypass_encoder(settings)
    before = {k: v.clone() for k, v in encoder.model.state_dict().items()}
    prepared = synthetic_prepared(1, per_view=2)
    vector = encoder.subject_features(prepared)
    assert vector.shape == (4 * 1280,) == (5120,) and vector.dtype == np.float32
    assert np.isfinite(vector).all()
    sagittal = encoder._view_features(prepared["views"]["sagittal"], 2)
    assert sagittal.shape == (2, 2560)
    assert np.allclose(vector[:2560], sagittal.mean(0), atol=1e-4)
    assert np.allclose(vector[2560:5120], sagittal.std(0), atol=1e-4)
    assert not encoder.model.training
    assert all(not p.requires_grad for p in encoder.model.parameters())
    assert isinstance(encoder.model.classifier, torch.nn.Identity)
    after = encoder.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert np.array_equal(vector, encoder.subject_features(prepared))


def test_wrong_embedding_dimension_is_rejected(settings):
    encoder = bypass_encoder(settings)
    encoder.dimension = 999
    with pytest.raises(ValueError):
        encoder.subject_features(synthetic_prepared(1, per_view=2))


def test_metadata_describes_frozen_recipe(settings):
    meta = bypass_encoder(settings).metadata()
    assert meta["backbone"] == "efficientnet_b0" and meta["dimension"] == 1280 and meta["image_size"] == 224
    assert meta["frozen"] is True and meta["pooling"] == "conv-map-spatial-mean-std"
    assert meta["weight_sha256"] == "0" * 64


def test_encoder_requires_verified_official_weights(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
    with pytest.raises(FileNotFoundError):
        deep.FrozenEncoder("efficientnet_b0", settings, allow_download=False)
    with pytest.raises(FileNotFoundError):
        deep.FrozenEncoder("efficientnet_b0", settings, weights_path=tmp_path / "missing.pth")
    fake = tmp_path / "fake.pth"
    fake.write_bytes(b"not official weights")
    with pytest.raises(ValueError):
        deep.FrozenEncoder("efficientnet_b0", settings, weights_path=fake)
    with pytest.raises(ValueError):
        deep.FrozenEncoder("resnet50", settings)


@pytest.fixture
def stubs(tmp_path, monkeypatch):
    weights = tmp_path / "w.bin"
    weights.write_bytes(b"weights-v1")
    calls = {"prepare": 0}

    def fake_prepare(path, settings, hash_voxels=False):
        calls["prepare"] += 1
        return synthetic_prepared(int(str(path).split("_")[-1]))

    monkeypatch.setattr(features, "prepare_subject", fake_prepare)
    rows = [{"subject_id": str(i), "source_sha256": f"sha{i}", "raw_path": f"raw_{i}"}
            for i in (1, 2, 3)]
    encoders = {"efficientnet_b0": StubEncoder("efficientnet_b0", weights, 4)}
    return weights, rows, encoders, calls


def test_extract_features_schema_and_saliency_columns(stubs, tmp_path, settings):
    _, rows, encoders, _ = stubs
    frame, metadata, schema = features.extract_features(rows, settings, tmp_path / "c", encoders=encoders,
                                                        backbones=("efficientnet_b0",))
    deep_cols = [c for c in frame if c.startswith("deep__efficientnet_b0__")]
    assert len(deep_cols) == 16
    assert len([c for c in frame if c.startswith("texture__")]) == 72
    assert len([c for c in frame if c.startswith("saliency__")]) == 12
    assert list(frame.columns[-12:])[0].startswith("saliency__")
    assert not any(c in ("age", "sex_female", "fiq") for c in frame.columns)
    assert list(frame.index) == ["1", "2", "3"]
    assert schema["columns"] == list(frame.columns) and metadata["efficientnet_b0"]["dimension"] == 4
    assert np.isfinite(frame.to_numpy()).all()


def test_cache_hit_is_identical_and_skips_extraction(stubs, tmp_path, settings):
    _, rows, encoders, calls = stubs
    kwargs = dict(encoders=encoders, backbones=("efficientnet_b0",))
    first, _, _ = features.extract_features(rows, settings, tmp_path / "c", **kwargs)
    assert calls["prepare"] == 3 and encoders["efficientnet_b0"].calls == 3
    second, _, _ = features.extract_features(rows, settings, tmp_path / "c", **kwargs)
    assert calls["prepare"] == 3 and encoders["efficientnet_b0"].calls == 3
    assert first.equals(second)


def test_cache_contains_no_labels_or_clinical_values(stubs, tmp_path, settings):
    _, rows, encoders, _ = stubs
    features.extract_features(rows, settings, tmp_path / "c", encoders=encoders, backbones=("efficientnet_b0",))
    with np.load(next((tmp_path / "c").glob("*.npz")), allow_pickle=False) as arrays:
        assert sorted(arrays.files) == ["cnn_sagittal", "efficientnet_b0", "saliency", "texture"]


def test_cache_invalidated_by_source_config_and_weight_changes(stubs, tmp_path, settings):
    weights, rows, encoders, calls = stubs
    kwargs = dict(backbones=("efficientnet_b0",))
    features.extract_features(rows, settings, tmp_path / "c", encoders=encoders, **kwargs)
    changed = [dict(rows[0], source_sha256="different")] + rows[1:]
    features.extract_features(changed, settings, tmp_path / "c", encoders=encoders, **kwargs)
    assert calls["prepare"] == 4
    features.extract_features(rows, dataclasses.replace(settings, n_slices=8), tmp_path / "c",
                              encoders=encoders, **kwargs)
    assert calls["prepare"] == 7
    other = tmp_path / "w2.bin"
    other.write_bytes(b"weights-v2")
    swapped = {"efficientnet_b0": StubEncoder("efficientnet_b0", other, 4)}
    features.extract_features(rows, settings, tmp_path / "c", encoders=swapped, **kwargs)
    assert calls["prepare"] == 10


def test_corrupted_cache_is_refused(stubs, tmp_path, settings):
    _, rows, encoders, _ = stubs
    kwargs = dict(encoders=encoders, backbones=("efficientnet_b0",))
    features.extract_features(rows, settings, tmp_path / "c", **kwargs)
    target = sorted((tmp_path / "c").glob("*.npz"))[0]
    target.write_bytes(target.read_bytes()[:-8] + b"corrupt!")
    with pytest.raises(ValueError):
        features.extract_features(rows, settings, tmp_path / "c", **kwargs)


def test_incomplete_cache_without_metadata_is_recomputed(stubs, tmp_path, settings):
    _, rows, encoders, calls = stubs
    kwargs = dict(encoders=encoders, backbones=("efficientnet_b0",))
    features.extract_features(rows, settings, tmp_path / "c", **kwargs)
    next((tmp_path / "c").glob("*.json")).unlink()
    features.extract_features(rows, settings, tmp_path / "c", **kwargs)
    assert calls["prepare"] == 4


def test_expired_deadline_and_duplicate_rows_fail(stubs, tmp_path, settings):
    _, rows, encoders, calls = stubs
    kwargs = dict(encoders=encoders, backbones=("efficientnet_b0",))
    with pytest.raises(TimeoutError):
        features.extract_features(rows, settings, tmp_path / "c", deadline=time.monotonic() - 1, **kwargs)
    assert calls["prepare"] == 0
    with pytest.raises(ValueError):
        features.extract_features(rows + [rows[0]], settings, tmp_path / "d", **kwargs)


def test_two_backbones_have_independent_column_blocks(stubs, tmp_path, settings):
    weights, rows, encoders, _ = stubs
    both = dict(encoders)
    both["efficientnet_b5"] = StubEncoder("efficientnet_b5", weights, 6)
    frame, metadata, _ = features.extract_features(
        rows, settings, tmp_path / "c", encoders=both, backbones=("efficientnet_b0", "efficientnet_b5"))
    assert len([c for c in frame if c.startswith("deep__efficientnet_b0__")]) == 16
    assert len([c for c in frame if c.startswith("deep__efficientnet_b5__")]) == 24
    assert metadata["efficientnet_b5"]["weight_sha256"] == sha256_file(weights)
