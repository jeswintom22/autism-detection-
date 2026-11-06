import argparse
import dataclasses
import importlib
import sys
import tempfile
from pathlib import Path

from . import artifacts

REPO_ROOT = artifacts.REPO_ROOT
DEFAULT_CONFIG = "configs/hybrid.json"
SMOKE_PER_CLASS = 8
SMOKE_BUDGET_HOURS = 1.0
CONFIG_FILE = "config.json"
STATE_FILE = "run_state.json"


class CliError(Exception):
    pass


def _core(name):
    try:
        return importlib.import_module(f"asd_training.{name}")
    except ModuleNotFoundError as error:
        if error.name == f"asd_training.{name}":
            raise CliError(f"core module asd_training.{name} is not available yet") from error
        raise


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _say(message):
    print(message, flush=True)


def _manifest_paths(run_dir, name):
    return run_dir / f"{name}_manifest.json", run_dir / f"{name}_manifest.csv"


def _write_manifest(run_dir, name, rows):
    json_path, csv_path = _manifest_paths(run_dir, name)
    artifacts.atomic_json(json_path, {"rows": rows})
    artifacts.atomic_csv(csv_path, rows)


def _load_manifest(run_dir, name):
    return artifacts.read_json(_manifest_paths(run_dir, name)[0])["rows"]


def _load_development_manifest(run_dir):
    return _load_manifest(run_dir, "development")


def _load_test_manifest(run_dir):
    return _load_manifest(run_dir, "test")


def _load_run_settings(run_dir):
    config_module = _core("config")
    path = run_dir / CONFIG_FILE
    if not path.is_file():
        raise CliError(f"{path} not found; run prepare first")
    return config_module.settings_from_dict(artifacts.read_json(path))


def _load_state(run_dir):
    path = run_dir / STATE_FILE
    if not path.is_file():
        raise CliError(f"{path} not found; run prepare first")
    return artifacts.read_json(path)


def _registry_path(settings, splits):
    return Path(settings.repo_root) / "runs" / "hybrid" / "protocols" / str(splits.get("cohort_hash")) / "splits.json"


def _verify_run(run_dir, settings, state, splits, check_test):
    config_module = _core("config")
    problems = []
    payload_hash = artifacts.stable_hash({k: v for k, v in splits.items() if k != "split_hash"})
    if splits.get("split_hash") != payload_hash:
        problems.append("split content does not match its declared hash")
    if set(splits.get("development_ids", [])) & set(splits.get("test_ids", [])):
        problems.append("development and test IDs overlap")
    if settings.purpose == "production":
        registry = _registry_path(settings, splits)
        if not registry.is_file():
            problems.append("immutable split registry entry is missing")
        elif artifacts.read_json(registry) != splits:
            problems.append("splits differ from the immutable registry")
    if config_module.config_hash(settings) != state.get("config_hash"):
        problems.append("config hash does not match prepared state")
    if splits.get("split_hash") != state.get("split_hash"):
        problems.append("split hash does not match prepared state")
    if splits.get("cohort_hash") != state.get("cohort_hash"):
        problems.append("cohort hash does not match prepared state")
    dev = _load_development_manifest(run_dir)
    if artifacts.stable_hash(dev) != state.get("development_manifest_hash"):
        problems.append("development manifest changed after prepare")
    if sorted(row["subject_id"] for row in dev) != sorted(splits["development_ids"]):
        problems.append("development manifest does not match split development_ids")
    if check_test:
        test = _load_test_manifest(run_dir)
        if artifacts.stable_hash(test) != state.get("test_manifest_hash"):
            problems.append("test manifest changed after prepare")
        if sorted(row["subject_id"] for row in test) != sorted(splits["test_ids"]):
            problems.append("test manifest does not match split test_ids")
    if problems:
        raise CliError("run integrity check failed: " + "; ".join(problems))
    return dev


def _existing_run_compatible(run_dir, settings):
    config_module = _core("config")
    path = run_dir / CONFIG_FILE
    if not path.is_file():
        others = [p for p in run_dir.iterdir()] if run_dir.is_dir() else []
        if others:
            raise CliError(f"{run_dir} is not empty and has no {CONFIG_FILE}; refusing to use it")
        return False
    existing = config_module.settings_from_dict(artifacts.read_json(path))
    if config_module.config_hash(existing) != config_module.config_hash(settings):
        raise CliError(f"{run_dir} was prepared with a different configuration; use a new run directory")
    return True


def _verify_source_drift(run_dir, settings, rows):
    try:
        _core("dataset").verify_sources(rows)
    except ValueError as error:
        raise CliError(f"source data changed since the audit: {error}") from error


def _check_registry(settings, splits):
    registry = Path(settings.repo_root) / "runs" / "hybrid" / "protocols" / splits["cohort_hash"] / "splits.json"
    artifacts.ensure_safe_output(registry, settings)
    if registry.is_file():
        existing = artifacts.read_json(registry)
        if existing.get("split_hash") != splits["split_hash"] or existing != splits:
            raise CliError(
                f"existing protocol registry {registry} does not match the newly created split; "
                "refusing to replace it"
            )
        _say(f"reusing immutable split registry {registry}")
        return existing
    artifacts.atomic_json(registry, splits)
    _say(f"created immutable split registry {registry}")
    return splits


def prepare(config_path, run_dir):
    config_module = _core("config")
    settings = config_module.load_config(_resolve(config_path))
    run_dir = artifacts.ensure_safe_output(_resolve(run_dir), settings)
    if settings.purpose != "production":
        raise CliError("prepare requires a production configuration; use smoke for smoke runs")
    resumed = _existing_run_compatible(run_dir, settings)
    state_path = run_dir / STATE_FILE
    if resumed and state_path.is_file():
        state = artifacts.read_json(state_path)
        splits = artifacts.read_json(run_dir / "splits.json")
        dev = _verify_run(run_dir, settings, state, splits, check_test=True)
        _verify_source_drift(run_dir, settings, dev + _load_test_manifest(run_dir))
        _say(f"run {run_dir} already prepared and verified (cohort {state['cohort_hash'][:12]})")
        return run_dir

    dataset = _core("dataset")
    splits_module = _core("splits")
    run_dir.mkdir(parents=True, exist_ok=True)
    _say("auditing dataset (reads every source volume; this can take a while)")
    rows, report = dataset.audit_dataset(settings)
    artifacts.atomic_json(run_dir / "audit.json", report)
    blocking = report.get("blocking_errors") or []
    if blocking:
        _say(f"audit found {len(blocking)} blocking error(s); see {run_dir / 'audit.json'}")
        for item in blocking[:20]:
            _say(f"  - {item}")
        raise CliError("blocking audit errors; no split was created")
    for warning in report.get("warnings") or []:
        _say(f"  warning: {warning}")
    _say(f"audit passed for {len(rows)} patients ({report.get('unique_persons', len(rows))} unique people)")

    splits = splits_module.create_splits(rows, settings)
    splits_module.validate_splits(rows, splits)
    splits = _check_registry(settings, splits)

    development_ids = set(splits["development_ids"])
    test_ids = set(splits["test_ids"])
    development = sorted((r for r in rows if r["subject_id"] in development_ids), key=lambda r: r["subject_id"])
    test = sorted((r for r in rows if r["subject_id"] in test_ids), key=lambda r: r["subject_id"])
    if len(development) != len(development_ids) or len(test) != len(test_ids):
        raise CliError("frozen split IDs do not match the audited cohort rows")

    artifacts.atomic_json(run_dir / CONFIG_FILE, settings.to_dict())
    artifacts.atomic_json(run_dir / "environment.json", artifacts.collect_environment())
    artifacts.atomic_csv(run_dir / "source_manifest.csv", rows)
    _write_manifest(run_dir, "development", development)
    _write_manifest(run_dir, "test", test)
    artifacts.atomic_json(run_dir / "splits.json", splits)
    artifacts.atomic_json(state_path, {
        "schema_version": 1,
        "purpose": settings.purpose,
        "config_hash": config_module.config_hash(settings),
        "cohort_hash": splits["cohort_hash"],
        "split_hash": splits["split_hash"],
        "development_manifest_hash": artifacts.stable_hash(development),
        "test_manifest_hash": artifacts.stable_hash(test),
        "n_development": len(development),
        "n_test": len(test),
    })
    _say(f"prepared {run_dir}: {len(development)} development / {len(test)} test patients")
    return run_dir


def train(run_dir):
    run_dir = _resolve(run_dir)
    settings = _load_run_settings(run_dir)
    artifacts.ensure_safe_output(run_dir, settings)
    state = _load_state(run_dir)
    splits = artifacts.read_json(run_dir / "splits.json")
    development = _verify_run(run_dir, settings, state, splits, check_test=False)
    training = _core("training")
    _say(f"training on {len(development)} development patients in {run_dir}")
    result = training.train_development(run_dir, settings, development, splits)
    _say("development training finished; test set was not scored")
    return result


def evaluate_test(run_dir, confirm_final):
    if not confirm_final:
        raise CliError("evaluate-test requires --confirm-final")
    run_dir = _resolve(run_dir)
    settings = _load_run_settings(run_dir)
    artifacts.ensure_safe_output(run_dir, settings)
    if settings.purpose != "production":
        raise CliError("smoke runs cannot evaluate the production test set")
    state = _load_state(run_dir)
    splits = artifacts.read_json(run_dir / "splits.json")
    if not (run_dir / "frozen_bundle_manifest.json").is_file():
        raise CliError("frozen_bundle_manifest.json not found; finish training first")
    _verify_run(run_dir, settings, state, splits, check_test=True)
    test_rows = _load_test_manifest(run_dir)
    evaluation = _core("evaluation")
    _say(f"final frozen evaluation of {len(test_rows)} test patients")
    return evaluation.evaluate_frozen(run_dir, settings, test_rows, splits)


def report(run_dir):
    from . import reporting

    path = reporting.render_report(_resolve(run_dir))
    _say(f"report written: {path}")
    return path


def paper(source_run, run_dir, aug_copies, finetune_overrides):
    """Paper-style slice-level CV; reuses the cohort manifests of a prepared run."""
    source_run = _resolve(source_run)
    settings = _load_run_settings(source_run)
    state = _load_state(source_run)
    splits = artifacts.read_json(source_run / "splits.json")
    _verify_run(source_run, settings, state, splits, check_test=True)
    rows = _load_manifest(source_run, "development") + _load_manifest(source_run, "test")
    if finetune_overrides:
        settings = dataclasses.replace(
            settings, finetune={**(settings.finetune or {}), **finetune_overrides})
    run_dir = artifacts.ensure_safe_output(_resolve(run_dir), settings)
    if artifacts._norm(run_dir) == artifacts._norm(source_run):
        raise CliError("paper run directory must differ from the source run")
    if artifacts._is_within(artifacts._norm(run_dir), artifacts._norm(source_run)):
        raise CliError("paper run directory must not be inside the source run")
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts.atomic_json(run_dir / CONFIG_FILE, settings.to_dict())
    artifacts.atomic_json(run_dir / "environment.json", artifacts.collect_environment())
    _say(f"paper-style slice-level evaluation over {len(rows)} patients in {run_dir}")
    _say("note: slices of the same patient share train/test partitions (paper protocol)")
    return _core("paper_eval").run(run_dir, settings, rows, aug_copies=aug_copies)


def run(config_path, run_dir):
    prepared = prepare(config_path, run_dir)
    train(prepared)
    return prepared


def _smoke_run_dir(path, settings):
    target = artifacts.ensure_safe_output(_resolve(path), settings)
    production_root = Path(settings.repo_root) / "runs" / "hybrid"
    probe = artifacts._norm(target)
    if artifacts._is_within(probe, production_root):
        raise CliError("smoke output must be an independent temporary directory, not under runs/hybrid")
    if not artifacts._is_within(probe, Path(tempfile.gettempdir())):
        raise CliError("smoke output must be inside the system temporary directory")
    return target


def smoke(source_run, run_dir):
    config_module = _core("config")
    splits_module = _core("splits")
    source_run = _resolve(source_run)
    source_settings = _load_run_settings(source_run)
    if source_settings.purpose != "production":
        raise CliError("smoke source run must be a prepared production run")
    state = _load_state(source_run)
    source_splits = artifacts.read_json(source_run / "splits.json")
    source_dev = _verify_run(source_run, source_settings, state, source_splits, check_test=False)

    settings = dataclasses.replace(
        source_settings,
        purpose="smoke",
        backbones=("efficientnet_b0",),
        outer_folds=2,
        inner_folds=2,
        n_estimators=10,
        search_iterations=1,
        soft_budget_hours=min(SMOKE_BUDGET_HOURS, source_settings.soft_budget_hours),
    )
    run_dir = _smoke_run_dir(run_dir, settings)
    if artifacts._norm(run_dir) == artifacts._norm(source_run):
        raise CliError("smoke run directory must differ from the source run")

    development_ids = set(source_splits["development_ids"])
    selected = []
    for label in (0, 1):
        candidates = [
            row for row in source_dev
            if row["label"] == label and row["subject_id"] in development_ids
        ]
        candidates.sort(key=lambda r: (int(r["subject_id"]) if str(r["subject_id"]).isdigit() else 0, r["subject_id"]))
        if len(candidates) < SMOKE_PER_CLASS:
            raise CliError(
                f"only {len(candidates)} development patients with label {label}; "
                f"need {SMOKE_PER_CLASS}; not borrowing from the test set"
            )
        selected.extend(candidates[:SMOKE_PER_CLASS])
    selected.sort(key=lambda r: r["subject_id"])

    if _existing_run_compatible(run_dir, settings):
        _say(f"resuming smoke run {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    splits = splits_module.create_development_folds(selected, settings)
    splits_module.validate_splits(selected, splits)
    if set(splits.get("test_ids") or []):
        raise CliError("smoke splits must not contain test IDs")
    artifacts.atomic_json(run_dir / CONFIG_FILE, settings.to_dict())
    artifacts.atomic_json(run_dir / "environment.json", artifacts.collect_environment())
    artifacts.atomic_json(run_dir / "splits.json", splits)
    artifacts.atomic_json(run_dir / "audit.json", {
        "schema_version": 2,
        "purpose": "smoke",
        "source_run": str(source_run),
        "blocking_errors": [],
    })
    _write_manifest(run_dir, "development", selected)
    artifacts.atomic_json(run_dir / STATE_FILE, {
        "schema_version": 1,
        "purpose": "smoke",
        "source_run": str(source_run),
        "config_hash": config_module.config_hash(settings),
        "cohort_hash": splits["cohort_hash"],
        "split_hash": splits["split_hash"],
        "development_manifest_hash": artifacts.stable_hash(selected),
        "n_development": len(selected),
    })
    training = _core("training")
    _say(f"smoke training on {len(selected)} development patients in {run_dir}")
    return training.train_development(run_dir, settings, selected, splits)


class _RejectSlice(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error("slice-level splitting is not supported; splits are always patient-level")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="asd_training",
        description="Patient-safe EfficientNet + texture + Random Forest pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--split", action=_RejectSlice, help=argparse.SUPPRESS)
        return p

    p = add("prepare", "audit data and lock the patient split; no test scoring")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--run-dir", required=True)

    p = add("smoke", "small development-only profile run from a prepared run")
    p.add_argument("--source-run", required=True)
    p.add_argument("--run-dir", required=True)

    p = add("train", "extract development features, select models, freeze bundles")
    p.add_argument("--run-dir", required=True)

    p = add("evaluate-test", "one locked evaluation of the frozen bundles")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--confirm-final", action="store_true")

    p = add("report", "render saved evidence into report.md and figures")
    p.add_argument("--run-dir", required=True)

    p = add("run", "prepare then train; never evaluates test")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--run-dir", required=True)

    p = sub.add_parser(
        "paper",
        help="paper-style slice-level CV (leaky by design; comparison only)")
    p.add_argument("--source-run", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--aug-copies", type=int, default=0,
                   help="near-duplicate augmented copies per slice before splitting")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra:
        parser.error(
            f"unrecognized or unsupported arguments: {' '.join(extra)} "
            "(legacy step3/run_all options are not supported by the new pipeline)"
        )
    try:
        if args.command == "prepare":
            prepare(args.config, args.run_dir)
        elif args.command == "smoke":
            smoke(args.source_run, args.run_dir)
        elif args.command == "train":
            train(args.run_dir)
        elif args.command == "evaluate-test":
            evaluate_test(args.run_dir, args.confirm_final)
        elif args.command == "report":
            report(args.run_dir)
        elif args.command == "run":
            run(args.config, args.run_dir)
        elif args.command == "paper":
            overrides = {k: v for k, v in (
                ("epochs", args.epochs), ("batch_size", args.batch_size),
                ("patience", args.patience)) if v is not None}
            paper(args.source_run, args.run_dir, args.aug_copies, overrides)
    except TimeoutError as error:
        print(f"incomplete: {error}; completed checkpoints are preserved, rerun the same command to resume",
              file=sys.stderr)
        return 3
    except CliError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (PermissionError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
