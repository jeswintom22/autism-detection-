"""One-off smoke driver for the paper-style slice-level evaluation."""
import dataclasses
import tempfile
from pathlib import Path


def main():
    from asd_training import artifacts
    from asd_training.config import settings_from_dict
    from asd_training.paper_eval import run

    RUN = Path("runs/hybrid/r4")
    settings = settings_from_dict(artifacts.read_json(RUN / "config.json"))
    rows = artifacts.read_json(RUN / "development_manifest.json")["rows"]
    per_class = {}
    for row in rows:
        per_class.setdefault(row["label"], []).append(row)
    selected = sorted(
        per_class[0][:8] + per_class[1][:8], key=lambda r: r["subject_id"])
    print(f"smoke cohort: {len(selected)} patients")

    settings = dataclasses.replace(
        settings,
        outer_folds=2,
        n_estimators=10,
        extract_workers=4,
        soft_budget_hours=2.0,
        finetune={**(settings.finetune or {}), "enabled": True,
                  "backbone": "efficientnet_b0", "epochs": 2, "batch_size": 4,
                  "patience": 2, "val_fraction": 0.2},
        batch_sizes={**settings.batch_sizes, "efficientnet_b0": 8},
    )
    run_dir = Path(tempfile.gettempdir()) / "asd_paper_smoke_aug"
    result = run(run_dir, settings, selected, aug_copies=1)
    print("families:", list(result["families"]))
    for fam, f in result["families"].items():
        print(f"  {fam}: fold accs {f['fold_accuracies']} "
              f"pooled {f['pooled_slice_metrics']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
