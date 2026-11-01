"""Average per-fold subject probabilities across independent runs that used
identical subject-level folds (e.g. --view axial / coronal / sagittal trained
with the same seed and dataset), then recompute metrics on the ensemble.

Each run writes results/val_probs.json (per-fold held-out subject ids, labels,
probabilities, tuned threshold). Because every run's fold-k models were trained
without the fold-k subjects, averaging their held-out predictions is a
legitimate ensemble evaluation - no leakage.

    python step3_classify.py --n-slices 9 --view axial    --models m_ax --results r_ax
    python step3_classify.py --n-slices 9 --view coronal  --models m_co --results r_co
    python step3_classify.py --n-slices 9 --view sagittal --models m_sa --results r_sa
    python ensemble_eval.py r_ax r_co r_sa
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def load_probs(results_dir):
    path = os.path.join(results_dir, "val_probs.json")
    if not os.path.exists(path):
        sys.exit(f"missing {path} - run step3_classify.py with this results dir")
    with open(path) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+",
                    help="results dirs from separate runs (>=2)")
    ap.add_argument("--out", default=None,
                    help="write ensemble metrics json here (default: stdout only)")
    args = ap.parse_args(argv)
    if len(args.dirs) < 2:
        ap.error("need at least 2 results dirs to ensemble")

    runs = {d: load_probs(d) for d in args.dirs}
    common_folds = sorted(set.intersection(*[set(r) for r in runs.values()]))
    if not common_folds:
        sys.exit("no shared folds across the runs")

    folds, cms = {}, []
    for fk in common_folds:
        maps = []
        thrs = []
        for d in args.dirs:
            fd = runs[d][fk]
            thrs.append(fd["threshold"])
            maps.append({s["file"]: (s["y"], s["p"]) for s in fd["subjects"]})
        common = set.intersection(*[set(m) for m in maps])
        if len(common) < len(maps[0]):
            print(f"  {fk}: {len(maps[0]) - len(common)} subjects missing in "
                  "some run - ensembling the intersection")
        if not common:
            print(f"  {fk}: no shared subjects - skipped")
            continue
        files = sorted(common)
        ys = np.array([maps[0][f][0] for f in files])
        ps = np.mean([[m[f][1] for m in maps] for f in files], axis=1)
        thr = float(np.mean(thrs))
        pred = (ps > thr).astype(int)
        try:
            auc = roc_auc_score(ys, ps)
        except ValueError:
            auc = float("nan")
        folds[fk] = {
            "accuracy": float(accuracy_score(ys, pred)),
            "precision": float(precision_score(ys, pred, zero_division=0)),
            "recall": float(recall_score(ys, pred, zero_division=0)),
            "f1": float(f1_score(ys, pred, zero_division=0)),
            "auc": float(auc),
            "threshold": thr,
            "n_subjects": len(files),
        }
        cms.append(confusion_matrix(ys, pred, labels=[0, 1]))
        print(f"{fk}: " + str({a: round(b, 4) for a, b in folds[fk].items()
                               if a != "n_subjects"}))

    if not folds:
        sys.exit("nothing to ensemble")
    overall = {
        mt: {"mean": float(np.nanmean([f[mt] for f in folds.values()])),
             "std": float(np.nanstd([f[mt] for f in folds.values()]))}
        for mt in ["accuracy", "precision", "recall", "f1", "auc"]
    }
    print(f"\nEnsemble of {len(args.dirs)} runs (subject-level):")
    for mt, s in overall.items():
        print(f"  {mt}: {s['mean']:.3f} +/- {s['std']:.3f}")

    out = {"runs": [os.path.abspath(d) for d in args.dirs],
           "fold_metrics": folds, "overall_metrics": overall}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=4)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
