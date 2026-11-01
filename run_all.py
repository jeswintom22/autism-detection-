"""Run the full pipeline: step1 -> step2 -> validation gate -> step3.

    python run_all.py                    # everything
    python run_all.py --skip-preprocess  # reuse existing preprocessed_dataset/
    python run_all.py --max 5 --folds 2 --epochs 1 --limit 10   # smoke test
"""
import argparse
import json
import os
import subprocess
import sys

import config

HERE = os.path.dirname(os.path.abspath(__file__))


def run_step(script, args):
    cmd = [sys.executable, os.path.join(HERE, script)] + args
    print(f"\n>>> {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"{script} failed with code {r.returncode}")


def count_nii(folder):
    if not os.path.isdir(folder):
        return -1
    n = 0
    for _, _, fs in os.walk(folder):
        n += sum(f.endswith((".nii", ".nii.gz")) for f in fs)
    return n


def validation_gate(preprocessed_dir):
    """Gate before training, mirroring the template's split_dataset checks."""
    problems = []
    counts = {}
    for class_name in config.LABELS:
        n = count_nii(os.path.join(preprocessed_dir, class_name))
        counts[class_name] = n
        if n < 0:
            problems.append(f"missing {preprocessed_dir}/{class_name}")
        elif n == 0:
            problems.append(f"empty {class_name}/ in {preprocessed_dir}")
    if not problems and os.path.exists(config.STEP2_SUMMARY):
        with open(config.STEP2_SUMMARY) as f:
            s = json.load(f)
        if s.get("failed"):
            print(f"gate warning: {s['failed']} subjects failed preprocessing")
    print("gate:", counts)
    if problems:
        sys.exit("validation gate failed:\n  " + "\n  ".join(problems))
    print("validation gate passed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--max", type=int, default=0, help="step2: max subjects/class")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch", type=int, default=config.BATCH)
    ap.add_argument("--limit", type=int, default=0, help="step3: max subjects/class")
    ap.add_argument("--raw", action="store_true", help="step3 on raw data (ablation)")
    ap.add_argument("--backbone", default=config.BACKBONE,
                    choices=["resnet101", "resnet50"])
    ap.add_argument("--n-slices", type=int, default=config.N_SLICES)
    ap.add_argument("--planes", type=int, default=1, choices=[1, 3])
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--smooth", type=float, default=config.LABEL_SMOOTHING)
    ap.add_argument("--bb-lr-mult", type=float, default=config.BACKBONE_LR_MULT)
    ap.add_argument("--no-augment", action="store_true",
                    help="step3: disable train augmentation (ablation)")
    ap.add_argument("--no-tta", action="store_true",
                    help="step3: disable flip test-time augmentation")
    ap.add_argument("--workers", type=int, default=4,
                    help="step3: dataloader workers for the train loader")
    ap.add_argument("--view", default="axial", choices=list(config.VIEW_AXIS),
                    help="step3: anatomical plane for planes=1")
    ap.add_argument("--slice-select", default=config.SLICE_SELECT,
                    choices=["spread", "area"],
                    help="step3: slice selection method")
    args = ap.parse_args(argv)

    if not args.skip_collect:
        run_step("step1_collect_data.py", [])
    if not args.skip_preprocess:
        s2 = ["--jobs", str(args.jobs)]
        if args.max:
            s2 += ["--max", str(args.max)]
        run_step("step2_preprocess.py", s2)
    validation_gate(config.PREPROCESSED_DIR)

    s3 = ["--folds", str(args.folds), "--epochs", str(args.epochs),
          "--batch", str(args.batch), "--backbone", args.backbone,
          "--n-slices", str(args.n_slices), "--planes", str(args.planes),
          "--lr", str(args.lr), "--smooth", str(args.smooth),
          "--bb-lr-mult", str(args.bb_lr_mult), "--jobs", str(args.jobs),
          "--view", args.view, "--slice-select", args.slice_select,
          "--workers", str(args.workers)]
    if args.no_augment:
        s3 += ["--no-augment"]
    if args.no_tta:
        s3 += ["--no-tta"]
    if args.limit:
        s3 += ["--limit", str(args.limit)]
    if args.raw:
        s3 += ["--raw"]
    run_step("step3_classify.py", s3)
    print("\nPipeline finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
