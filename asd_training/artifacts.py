import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PREFIXES = ("saved_models", "results")
LEGACY_NAMES = ("slice_cache", "preprocessed_dataset")
HASH_CHUNK = 1 << 20


def _norm(path):
    return Path(os.path.normcase(os.path.abspath(os.path.realpath(str(path)))))


def _is_within(path, parent):
    path, parent = _norm(path), _norm(parent)
    return path == parent or parent in path.parents


def _legacy_roots(repo_root):
    repo_root = Path(repo_root)
    roots = []
    if repo_root.is_dir():
        for child in repo_root.iterdir():
            name = child.name
            if name.startswith(LEGACY_PREFIXES) or name in LEGACY_NAMES:
                roots.append(child)
    for name in LEGACY_NAMES:
        roots.append(repo_root / name)
    for prefix in LEGACY_PREFIXES:
        roots.append(repo_root / prefix)
    return roots


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False, default=_json_default,
    )


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def stable_hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _global_roots():
    return [REPO_ROOT / "runs" / "hybrid", Path(tempfile.gettempdir())]


def ensure_allowed_output(path):
    target = _norm(path)
    roots = _global_roots()
    if any(target == _norm(root) for root in roots):
        raise PermissionError(f"output path must be a subdirectory, not an allowed root itself: {path}")
    if not any(_is_within(target, root) for root in roots):
        raise PermissionError(f"output path must be under {roots[0]} or a temporary directory: {path}")
    if _is_within(target, REPO_ROOT) and not _is_within(target, roots[0]):
        raise PermissionError(f"output path must not be inside the repository: {path}")
    return target


def _guard_target(path, suffixes):
    target = Path(path)
    if target.suffix.lower() not in suffixes:
        raise ValueError(f"refusing to write {target}: expected one of {suffixes}")
    if target.is_dir():
        raise IsADirectoryError(f"refusing to replace directory {target}")
    resolved = ensure_allowed_output(target)
    for legacy in _legacy_roots(REPO_ROOT):
        if _is_within(resolved, legacy):
            raise PermissionError(f"refusing to write into legacy path: {target}")
    return target


def _atomic_write(target, writer, mode, **open_kwargs):
    target.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, temp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle_fd, mode, **open_kwargs) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def atomic_json(path, data):
    target = _guard_target(path, (".json",))
    text = json.dumps(
        data, indent=2, sort_keys=True, allow_nan=False, default=_json_default
    )

    def writer(handle):
        handle.write(text)
        handle.write("\n")

    _atomic_write(target, writer, "w", encoding="utf-8", newline="\n")


def atomic_text(path, text):
    target = _guard_target(path, (".md", ".txt"))

    def writer(handle):
        handle.write(text)

    _atomic_write(target, writer, "w", encoding="utf-8", newline="\n")


def _csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return _canonical(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_csv(path, rows, fieldnames=None):
    target = _guard_target(path, (".csv",))
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    fieldnames = list(fieldnames)

    def writer(handle):
        out = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        out.writeheader()
        for row in rows:
            out.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})

    _atomic_write(target, writer, "w", encoding="utf-8", newline="")


def ensure_safe_output(path, settings):
    target = _norm(path)
    repo_root = Path(settings.repo_root)
    for raw in (settings.control_dir, settings.treatment_dir):
        if _is_within(target, raw):
            raise PermissionError(f"output path is inside raw data directory: {path}")
    for legacy in _legacy_roots(repo_root):
        if _is_within(target, legacy):
            raise PermissionError(f"output path is inside legacy artifact path: {path}")
    allowed_roots = [repo_root / "runs" / "hybrid", Path(tempfile.gettempdir())]
    if any(target == _norm(root) for root in allowed_roots):
        raise PermissionError(f"output path must be a subdirectory, not an allowed root itself: {path}")
    if not any(_is_within(target, root) for root in allowed_roots):
        raise PermissionError(
            f"output path must be under {allowed_roots[0]} or a temporary directory: {path}"
        )
    if _is_within(target, repo_root) and not _is_within(target, allowed_roots[0]):
        raise PermissionError(f"output path must not be inside the repository: {path}")
    return Path(os.path.abspath(str(path)))


def collect_environment():
    import platform
    import sys
    from importlib import metadata

    packages = {}
    for name in (
        "numpy", "scipy", "pandas", "scikit-learn", "scikit-image", "nibabel",
        "torch", "torchvision", "joblib", "matplotlib", "pytest",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "cuda": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda"] = {
                "version": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            }
    except Exception as error:
        info["cuda"] = {"error": f"{type(error).__name__}: {error}"}
    return info
