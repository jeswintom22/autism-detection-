import numpy as np
import pytest

from asd_training import finetune
from asd_training.config import Settings


def _settings(**overrides):
    base = dict(
        views=("sagittal",), finetune={"enabled": True, "backbone": "efficientnet_b0",
                                     "epochs": 1, "batch_size": 2, "unfreeze_blocks": 1,
                                     "patience": 1, "val_fraction": 0.25},
        batch_sizes={"efficientnet_b0": 4, "efficientnet_b5": 2},
    )
    base.update(overrides)
    return Settings(**base)


def _patients(seed=0, n_per_class=3, slices=2, size=24):
    rng = np.random.default_rng(seed)
    patients, labels = [], {}
    for value in (0, 1):
        for i in range(n_per_class):
            sid = f"{value}{i}"
            views = {"sagittal": rng.random((slices, size, size), dtype=np.float32)}
            patients.append((sid, views))
            labels[sid] = value
    return patients, labels


def test_finetune_checkpoint_and_embeddings(tmp_path):
    settings = _settings()
    patients, labels = _patients()
    encoder = finetune.FineTunedEncoder(settings, "efficientnet_b0")
    checkpoint = tmp_path / "encoder.pt"
    meta = encoder.finetune(patients, labels, settings.finetune, checkpoint,
                            deadline=__import__("time").monotonic() + 600)
    assert checkpoint.is_file() and meta["sha256"]
    assert set(meta["train_ids"]) | set(meta["val_ids"]) == set(labels)
    assert not set(meta["train_ids"]) & set(meta["val_ids"])
    loaded = finetune.FineTunedEncoder.load(checkpoint, settings)
    vector = loaded.subject_features(patients[0][1])
    assert vector.shape == (2 * loaded.embedding_dim,) and np.isfinite(vector).all()
    loaded.close()
    encoder.close()


def test_finetune_split_is_patient_disjoint_and_complete():
    settings = _settings()
    _, labels = _patients()
    train, val = finetune.FineTunedEncoder._stratified_split(
        sorted(labels), labels, 0.34, settings.seed)
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(labels)
    assert {labels[s] for s in train} == {0, 1} and {labels[s] for s in val} == {0, 1}


def test_finetune_rejects_label_patient_mismatch(tmp_path):
    settings = _settings()
    patients, labels = _patients(n_per_class=2)
    encoder = finetune.FineTunedEncoder(settings, "efficientnet_b0")
    with pytest.raises(ValueError):
        encoder.finetune(patients, {k: v for k, v in list(labels.items())[:-1]},
                         settings.finetune, tmp_path / "e.pt",
                         deadline=__import__("time").monotonic() + 60)
    encoder.close()
