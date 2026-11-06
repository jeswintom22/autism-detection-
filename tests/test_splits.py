import copy
import dataclasses
import hashlib

import pytest

from asd_training import artifacts, cli, config, splits as splits_module
from asd_training.config import Settings


def fp(prefix, i):
    return hashlib.sha256(f"{prefix}{i}".encode()).hexdigest()


def make_rows(per_cell=20, sites=("A", "B")):
    rows, n = [], 0
    for site in sites:
        for label in (0, 1):
            for _ in range(per_cell):
                n += 1
                rows.append({"subject_id": str(100 + n), "label": label, "site_id": site,
                             "source_sha256": fp("s", n), "voxel_sha256": fp("v", n)})
    return rows


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def rows():
    return make_rows()


def test_complete_disjoint_coverage_and_expected_sizes(rows, settings):
    s = splits_module.create_splits(rows, settings)
    dev, test = set(s["development_ids"]), set(s["test_ids"])
    assert not dev & test and dev | test == {r["subject_id"] for r in rows}
    assert len(test) == 16 and len(dev) == 64
    assert len(s["outer_folds"]) == 5 and all(len(f["inner_folds"]) == 3 for f in s["outer_folds"])
    assert s["purpose"] == "production" and s["seed"] == 42
    by_id = {r["subject_id"]: r for r in rows}
    assert {by_id[i]["label"] for i in test} == {0, 1}
    splits_module.validate_splits(rows, s)


def test_outer_and_inner_folds_are_patient_grouped(rows, settings):
    s = splits_module.create_splits(rows, settings)
    dev = set(s["development_ids"])
    validated = []
    for fold in s["outer_folds"]:
        train, valid = set(fold["train_ids"]), set(fold["validation_ids"])
        assert not train & valid and train | valid == dev
        validated.extend(valid)
        inner_valid = []
        for inner in fold["inner_folds"]:
            itrain, ivalid = set(inner["train_ids"]), set(inner["validation_ids"])
            assert not itrain & ivalid and itrain | ivalid == train
            assert not ivalid & valid
            inner_valid.extend(ivalid)
        assert sorted(inner_valid) == sorted(train)
    assert sorted(validated) == sorted(dev)


def test_split_is_repeatable_and_independent_of_input_order(rows, settings):
    first = splits_module.create_splits(rows, settings)
    assert splits_module.create_splits(list(reversed(rows)), settings) == first
    assert splits_module.create_splits(copy.deepcopy(rows), settings)["split_hash"] == first["split_hash"]


def test_cohort_change_changes_identity(rows, settings):
    first = splits_module.create_splits(rows, settings)
    changed = copy.deepcopy(rows)
    changed[0]["voxel_sha256"] = fp("other", 0)
    second = splits_module.create_splits(changed, settings)
    assert second["cohort_hash"] != first["cohort_hash"]


def test_seed_cannot_be_changed():
    with pytest.raises(ValueError):
        config.settings_from_dict({**Settings().to_dict(), "seed": 7})


def test_slices_inherit_patient_assignment(rows, settings):
    s = splits_module.create_splits(rows, settings)
    slices = [(r["subject_id"], k) for r in rows for k in range(27)]
    dev, test = set(s["development_ids"]), set(s["test_ids"])
    dev_slices = [x for x in slices if x[0] in dev]
    test_slices = [x for x in slices if x[0] in test]
    assert len(dev_slices) + len(test_slices) == len(slices)
    assert not {x[0] for x in dev_slices} & {x[0] for x in test_slices}
    for fold in s["outer_folds"]:
        train = set(fold["train_ids"])
        assert not {x[0] for x in dev_slices if x[0] in train} & set(fold["validation_ids"])


def test_validate_rejects_tampering(rows, settings):
    base = splits_module.create_splits(rows, settings)

    def rehash(payload):
        payload = dict(payload)
        payload.pop("split_hash")
        payload["split_hash"] = artifacts.stable_hash(payload)
        return payload

    moved = copy.deepcopy(base)
    moved["outer_folds"][0]["validation_ids"].append(moved["outer_folds"][0]["train_ids"][0])
    with pytest.raises(ValueError):
        splits_module.validate_splits(rows, rehash(moved))

    stale = copy.deepcopy(base)
    stale["outer_folds"][0]["validation_ids"], stale["outer_folds"][1]["validation_ids"] = (
        stale["outer_folds"][1]["validation_ids"], stale["outer_folds"][0]["validation_ids"])
    with pytest.raises(ValueError):
        splits_module.validate_splits(rows, stale)

    overlap = copy.deepcopy(base)
    overlap["test_ids"].append(overlap["development_ids"][0])
    with pytest.raises(ValueError):
        splits_module.validate_splits(rows, rehash(overlap))

    incomplete = copy.deepcopy(base)
    incomplete["test_ids"] = incomplete["test_ids"][:-1]
    with pytest.raises(ValueError):
        splits_module.validate_splits(rows, rehash(incomplete))

    drift = copy.deepcopy(rows)
    drift[0]["voxel_sha256"] = fp("other", 0)
    with pytest.raises(ValueError):
        splits_module.validate_splits(drift, base)


def test_shared_fingerprints_and_duplicate_ids_are_refused(rows, settings):
    shared = copy.deepcopy(rows)
    shared[1]["voxel_sha256"] = shared[0]["voxel_sha256"]
    with pytest.raises(ValueError):
        splits_module.create_splits(shared, settings)
    dup = copy.deepcopy(rows)
    dup[1]["subject_id"] = dup[0]["subject_id"]
    with pytest.raises(ValueError):
        splits_module.create_splits(dup, settings)


def test_inadequate_strata_raise_explicit_errors(settings):
    thin = make_rows(per_cell=2)
    with pytest.raises(ValueError, match="stratum"):
        splits_module.create_splits(thin, settings)
    one_class = [r for r in make_rows() if r["label"] == 0]
    with pytest.raises(ValueError):
        splits_module.create_splits(one_class, settings)
    imbalanced = [r for r in make_rows() if r["label"] == 0][:30]
    imbalanced += [r for r in make_rows() if r["label"] == 1][:4]
    with pytest.raises(ValueError, match="stratum"):
        splits_module.create_splits(imbalanced, settings)


def test_purpose_gates_production_and_smoke_creators(rows, settings):
    smoke = dataclasses.replace(settings, purpose="smoke", outer_folds=2, inner_folds=2)
    with pytest.raises(ValueError):
        splits_module.create_splits(rows, smoke)
    with pytest.raises(ValueError):
        splits_module.create_development_folds(rows, settings)


def test_development_folds_have_no_test_ids(settings):
    smoke = dataclasses.replace(settings, purpose="smoke", outer_folds=2, inner_folds=2)
    rows = make_rows(per_cell=8, sites=("NYU",))
    s = splits_module.create_development_folds(rows, smoke)
    assert s["test_ids"] == [] and s["purpose"] == "smoke"
    assert set(s["development_ids"]) == {r["subject_id"] for r in rows}
    assert len(s["outer_folds"]) == 2
    splits_module.validate_splits(rows, s)


def test_registry_refuses_replacement_and_reuses_identical(tmp_path, rows):
    settings = Settings(repo_root=tmp_path / "repo", control_dir=tmp_path / "c", treatment_dir=tmp_path / "t")
    s = splits_module.create_splits(rows, settings)
    first = cli._check_registry(settings, s)
    assert first == s
    assert cli._check_registry(settings, s) == s
    registry = settings.repo_root / "runs" / "hybrid" / "protocols" / s["cohort_hash"] / "splits.json"
    mutated = copy.deepcopy(s)
    mutated["test_ids"], mutated["development_ids"] = mutated["development_ids"][:1], mutated["test_ids"]
    mutated.pop("split_hash")
    mutated["split_hash"] = artifacts.stable_hash(mutated)
    with pytest.raises(cli.CliError):
        cli._check_registry(settings, mutated)
    assert artifacts.read_json(registry) == s


def test_person_group_never_splits_and_tampering_is_refused(rows, settings):
    pair = rows[:2]
    pair[0]["person_id"] = pair[0]["subject_id"]
    pair[1]["person_id"] = pair[0]["subject_id"]
    pair[1]["source_sha256"] = pair[0]["source_sha256"]
    pair[1]["voxel_sha256"] = pair[0]["voxel_sha256"]
    s = splits_module.create_splits(rows, settings)
    dev, test = set(s["development_ids"]), set(s["test_ids"])
    both = {pair[0]["subject_id"], pair[1]["subject_id"]}
    assert both <= dev or both <= test
    for fold in s["outer_folds"]:
        assert both <= set(fold["train_ids"]) or both <= set(fold["validation_ids"])
    splits_module.validate_splits(rows, s)

    bad = copy.deepcopy(s)
    member = pair[0]["subject_id"]
    side = "development_ids" if member in dev else "test_ids"
    other = "test_ids" if side == "development_ids" else "development_ids"
    bad[side] = [sid for sid in bad[side] if sid != member]
    bad[other] = sorted(bad[other] + [member], key=int)
    bad.pop("split_hash")
    bad["split_hash"] = artifacts.stable_hash(bad)
    with pytest.raises(ValueError, match="person"):
        splits_module.validate_splits(rows, bad)
