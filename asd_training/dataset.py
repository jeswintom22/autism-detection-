import re
from collections import Counter, defaultdict
from pathlib import Path

from nibabel.filebasedimages import ImageFileError

from .artifacts import sha256_file
from .preprocessing import prepare_subject


RAW_NAME = re.compile(r"^(\d+)_skull_stripped\.nii(?:\.gz)?$")


def normalize_subject_id(value):
    value = str(value).strip()
    if not re.fullmatch(r"\d+", value):
        raise ValueError(f"Invalid numeric patient ID: {value!r}")
    return str(int(value))


def audit_dataset(settings):
    report = {"schema_version": 2, "blocking_errors": [], "failures": [], "patients": 0}
    errors = report["blocking_errors"]
    inputs = []
    for class_name, label, folder in (
        ("control", 0, settings.control_dir), ("treatment", 1, settings.treatment_dir),
    ):
        if not folder.is_dir():
            errors.append(f"Missing {class_name} source directory: {folder}")
            continue
        paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.name.endswith((".nii", ".nii.gz")))
        if not paths:
            errors.append(f"No MRI scans in {folder}")
        for path in paths:
            match = RAW_NAME.fullmatch(path.name)
            if match is None:
                errors.append(f"Unexpected raw filename: {path}")
                continue
            inputs.append((normalize_subject_id(match.group(1)), class_name, label, path.resolve()))
    duplicates = {sid for sid, count in Counter(item[0] for item in inputs).items() if count > 1}
    errors.extend(f"Duplicate patient ID across source scans/classes: {sid}" for sid in sorted(duplicates))
    rows = []
    for index, (sid, class_name, label, path) in enumerate(inputs, 1):
        try:
            if sid in duplicates:
                continue
            stat = path.stat()
            digest = sha256_file(path)
            prepared = prepare_subject(path, settings, hash_voxels=True)
            geometry = prepared["qc"]
            final_stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
                raise ValueError("Source scan changed during audit")
            rows.append({
                "subject_id": sid, "label": label, "class_name": class_name, "raw_path": str(path),
                "source_sha256": digest,
                "voxel_sha256": geometry["voxel_sha256"], "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns, "geometry": geometry,
            })
            del prepared
        except (ValueError, OSError, RuntimeError, TypeError, ImageFileError, EOFError) as exc:
            failure = {"subject_id": sid, "path": str(path), "error": str(exc)}
            report["failures"].append(failure)
            errors.append(f"Patient {sid}: {exc}")
        if index == 1 or index % 20 == 0 or index == len(inputs):
            print(f"audit: {index}/{len(inputs)} scans, {len(errors)} blocking errors", flush=True)
    parent = {row["subject_id"]: row["subject_id"] for row in rows}

    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    for field in ("source_sha256", "voxel_sha256"):
        seen = {}
        for row in rows:
            fingerprint = row[field]
            if fingerprint in seen:
                root_a, root_b = find(seen[fingerprint]), find(row["subject_id"])
                lo = min((root_a, root_b), key=int)
                parent[max((root_a, root_b), key=int)] = lo
                seen[fingerprint] = lo
            else:
                seen[fingerprint] = find(row["subject_id"])
    person_groups = defaultdict(list)
    for row in rows:
        person_groups[find(row["subject_id"])].append(row)
    report["duplicate_groups"] = []
    report["warnings"] = []
    for group in person_groups.values():
        ids = sorted((m["subject_id"] for m in group), key=int)
        shared = [field for field in ("source_sha256", "voxel_sha256")
                  if len({m[field] for m in group}) == 1]
        if len(group) > 1 and {m["label"] for m in group} != {group[0]["label"]}:
            for field in shared:
                errors.append(f"Duplicate {field} with conflicting identity: {', '.join(ids)}")
        if len(group) > 1:
            report["duplicate_groups"].append({"subject_ids": ids, "person_id": ids[0]})
            report["warnings"].append(
                f"Duplicate {'/'.join(shared)} grouped as one patient {ids[0]}: {', '.join(ids)}"
            )
        for member in group:
            member["person_id"] = ids[0]
    rows.sort(key=lambda row: int(row["subject_id"]))
    report.update({
        "patients": len(rows), "unique_persons": len(person_groups), "source_scan_count": len(inputs),
        "class_counts": dict(Counter(row["class_name"] for row in rows)),
    })
    return rows, report


def verify_sources(rows):
    for row in rows:
        path = Path(row["raw_path"])
        if not path.is_file() or sha256_file(path) != row["source_sha256"]:
            raise ValueError(f"Source changed since audit for patient {row['subject_id']}")
