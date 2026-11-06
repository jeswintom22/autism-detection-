from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .artifacts import stable_hash


def _person(row):
    return str(row.get("person_id") or row["subject_id"])


def _ordered(rows):
    rows = sorted(rows, key=lambda row: int(row["subject_id"]))
    if len({row["subject_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate patient IDs cannot be split")
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Both control and ASD labels are required")
    owner = {}
    labels = defaultdict(set)
    for row in rows:
        labels[_person(row)].add(row["label"])
        for field in ("source_sha256", "voxel_sha256"):
            fingerprint = row[field]
            if fingerprint in owner and owner[fingerprint] != _person(row):
                raise ValueError(f"Duplicate patient content across different people: {field}")
            owner[fingerprint] = _person(row)
    if any(len(group) > 1 for group in labels.values()):
        raise ValueError("Conflicting diagnosis labels inside one patient group")
    return rows


def _person_index(rows):
    members = defaultdict(list)
    for row in rows:
        members[_person(row)].append(row["subject_id"])
    persons = sorted(members, key=lambda pid: min(int(sid) for sid in members[pid]))
    mapping = {row["subject_id"]: row for row in rows}
    strata = np.asarray([
        str(mapping[members[pid][0]]["label"]) for pid in persons
    ])
    return persons, members, strata


def _expand(persons, members, indices):
    return sorted((sid for i in indices for sid in members[persons[i]]), key=int)


def _strata_guard(strata, folds):
    if not len(strata) or min(Counter(strata).values()) < folds:
        raise ValueError(f"Insufficient label stratum support for {folds} patient folds")


def _group_folds(rows, count, seed):
    persons, members, strata = _person_index(rows)
    _strata_guard(strata, count)
    mapping = {row["subject_id"]: row for row in rows}
    splitter = StratifiedKFold(n_splits=count, shuffle=True, random_state=seed)
    output = []
    for number, (train, valid) in enumerate(splitter.split(np.zeros(len(persons)), strata), 1):
        train_ids, valid_ids = _expand(persons, members, train), _expand(persons, members, valid)
        if ({mapping[sid]["label"] for sid in train_ids} != {0, 1}
                or {mapping[sid]["label"] for sid in valid_ids} != {0, 1}):
            raise ValueError("A patient fold is missing one diagnosis class")
        output.append({"fold": number, "train_ids": train_ids, "validation_ids": valid_ids})
    return output


def _cohort_hash(rows):
    return stable_hash([{
        key: (_person(row) if key == "person_id" else row[key])
        for key in ("subject_id", "person_id", "label", "source_sha256", "voxel_sha256")
    } for row in rows])


def _finish(rows, settings, development, test):
    development, test = sorted(development, key=int), sorted(test, key=int)
    mapping = {row["subject_id"]: row for row in rows}
    dev_rows = [mapping[sid] for sid in development]
    outer = _group_folds(dev_rows, settings.outer_folds, settings.seed)
    for fold in outer:
        fold["inner_folds"] = _group_folds(
            [mapping[sid] for sid in fold["train_ids"]],
            settings.inner_folds, settings.seed + fold["fold"],
        )
    output = {
        "schema_version": settings.schema_version, "purpose": settings.purpose,
        "cohort_hash": _cohort_hash(rows), "seed": settings.seed,
        "development_ids": development, "test_ids": test, "outer_folds": outer,
        "final_inner_folds": _group_folds(dev_rows, settings.inner_folds, 4242),
    }
    output["split_hash"] = stable_hash(output)
    validate_splits(rows, output)
    return output


def create_splits(rows, settings):
    if settings.purpose != "production":
        raise ValueError("Production holdout creation requires production purpose")
    rows = _ordered(rows)
    persons, members, strata = _person_index(rows)
    _strata_guard(strata, 2)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=settings.test_size, random_state=settings.seed)
    train, test = next(splitter.split(np.zeros(len(persons)), strata))
    return _finish(
        rows, settings, _expand(persons, members, train), _expand(persons, members, test),
    )


def create_development_folds(rows, settings):
    if settings.purpose != "smoke":
        raise ValueError("Development-only folds require explicit smoke purpose")
    rows = _ordered(rows)
    return _finish(rows, settings, [row["subject_id"] for row in rows], [])


def validate_splits(rows, splits):
    rows = _ordered(rows)
    mapping = {row["subject_id"]: row for row in rows}
    all_ids = set(mapping)
    development, test = set(splits["development_ids"]), set(splits["test_ids"])
    if len(development) != len(splits["development_ids"]) or len(test) != len(splits["test_ids"]):
        raise ValueError("Repeated patient membership in a partition")
    if development & test or development | test != all_ids:
        raise ValueError("Patient holdout overlap or incomplete coverage")
    if {_person(mapping[sid]) for sid in development} & {_person(mapping[sid]) for sid in test}:
        raise ValueError("One person's scans split across development and test")
    if splits.get("cohort_hash") != _cohort_hash(rows):
        raise ValueError("Cohort fingerprint drift")
    expected_hash = stable_hash({k: v for k, v in splits.items() if k != "split_hash"})
    if splits.get("split_hash") != expected_hash:
        raise ValueError("Split content fingerprint mismatch")
    if splits.get("purpose") == "production" and not test:
        raise ValueError("Production requires a locked patient test")
    if test and {mapping[sid]["label"] for sid in test} != {0, 1}:
        raise ValueError("Patient test partition must contain both diagnosis classes")

    _validate_folds(mapping, splits["outer_folds"], development)
    _validate_folds(mapping, splits["final_inner_folds"], development)


def _validate_folds(mapping, folds, scope):
    seen = []
    if len(folds) < 2:
        raise ValueError("At least two patient folds are required")
    for fold in folds:
        train, valid = set(fold["train_ids"]), set(fold["validation_ids"])
        if len(train) != len(fold["train_ids"]) or len(valid) != len(fold["validation_ids"]):
            raise ValueError("Duplicate patient within a fold")
        if train & valid or train | valid != scope:
            raise ValueError("Patient leakage or incomplete fold membership")
        if {_person(mapping[sid]) for sid in train} & {_person(mapping[sid]) for sid in valid}:
            raise ValueError("One person's scans split across a fold")
        for ids in (train, valid):
            if {mapping[sid]["label"] for sid in ids} != {0, 1}:
                raise ValueError("Patient fold missing a diagnosis class")
        for field in ("source_sha256", "voxel_sha256"):
            if {mapping[sid][field] for sid in train} & {mapping[sid][field] for sid in valid}:
                raise ValueError("Image content leakage across folds")
        seen.extend(valid)
        if "inner_folds" in fold:
            _validate_folds(mapping, fold["inner_folds"], train)
    if Counter(seen) != Counter(scope):
        raise ValueError("Each patient must validate exactly once in a CV layer")


def validate_development_splits(rows, splits):
    rows = _ordered(rows)
    mapping = {row["subject_id"]: row for row in rows}
    development, test = set(splits["development_ids"]), set(splits["test_ids"])
    if set(mapping) != development or development & test or len(development) != len(splits["development_ids"]):
        raise ValueError("Invalid development-only patient membership")
    if splits["split_hash"] != stable_hash({k: v for k, v in splits.items() if k != "split_hash"}):
        raise ValueError("Patient fold payload changed after preparation")
    _validate_folds(mapping, splits["outer_folds"], development)
    _validate_folds(mapping, splits["final_inner_folds"], development)
