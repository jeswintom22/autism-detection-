"""Step 4 - fuse held-out CNN probabilities with ABIDE phenotypic data.

Inputs:
  * one or more step3 results dirs (val_probs.json; several dirs = view
    ensemble, probabilities averaged per subject)
  * ABIDE I phenotypic CSV (auto-downloaded), joined on numeric subject id

Per outer fold k a logistic-regression stacker is fit on the subjects of the
OTHER folds - using their out-of-fold CNN probability + age/sex/FIQ(/site) -
and evaluated on fold k. Imputation, scaling and the decision threshold are
all fit on the training subjects only, so fold-k labels never touch fold-k
predictions. Reports CNN-only, phenotype-only and fused metrics.

    python step4_fuse.py results_ax results_co results_sa
    python step4_fuse.py results_ax --no-site --no-fiq
    python step4_fuse.py results_ax results_co results_sa --roi --no-site
(--roi needs step4a_fetch_freesurfer.py first)
"""
import argparse
import json
import os
import re
import sys
import urllib.request

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

import config

PHENO_URL = ("https://s3.amazonaws.com/fcp-indi/data/Projects/"
             "ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv")


def load_pheno(path):
    if not os.path.exists(path):
        print(f"downloading phenotypic CSV -> {path}")
        urllib.request.urlretrieve(PHENO_URL, path)
    df = pd.read_csv(path).set_index("SUB_ID")
    df["FIQ"] = df["FIQ"].where(df["FIQ"] > 0)          # -9999 = missing
    return df


def load_cnn(dirs):
    """{fold: {sub_id: (y, mean_p)}} averaged over runs."""
    runs = []
    for d in dirs:
        p = os.path.join(d, "val_probs.json")
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
        with open(p) as f:
            runs.append(json.load(f))
    out = {}
    for fk in sorted(set.intersection(*[set(r) for r in runs])):
        maps = [{int(re.match(r"(\d+)", s["file"]).group(1)): (s["y"], s["p"])
                 for s in r[fk]["subjects"]} for r in runs]
        common = set.intersection(*[set(m) for m in maps])
        out[fk] = {i: (maps[0][i][0], float(np.mean([m[i][1] for m in maps])))
                   for i in common}
    return out


def features(df, ids, p_cnn, use_cnn, use_pheno, use_site, use_fiq, sites,
             roi=None):
    cols = []
    if roi is not None:
        r = roi.reindex(ids).to_numpy(float)
        cols += list(r.T) + [np.isnan(r).all(1).astype(float)]
    if use_cnn:
        p = np.clip(p_cnn, 1e-4, 1 - 1e-4)
        cols.append(np.log(p / (1 - p)))
    if use_pheno:
        sub = df.loc[ids]
        age = sub["AGE_AT_SCAN"].to_numpy(float)
        cols += [age, np.log(age), (sub["SEX"].to_numpy() == 2).astype(float)]
        if use_fiq:
            fiq = sub["FIQ"].to_numpy(float)
            cols += [fiq, np.isnan(fiq).astype(float)]
        if use_site:
            s = sub["SITE_ID"].to_numpy()
            cols += [(s == site).astype(float) for site in sites]
    return np.column_stack(cols)


from step3_classify import best_threshold  # noqa: E402


def icv_normalise(roi):
    """Volumes / areas as a fraction of intracranial volume (head size)."""
    icv = next((roi[c] for c in roi if c.lower().endswith(("_icv", "_etiv"))),
               None)
    if icv is None:
        return roi
    roi = roi.copy()
    for c in roi:
        if c.startswith(("vol_", "gmv_", "area_")):
            roi[c] = roi[c] / icv
    return roi


def site_harmonise(roi, df, train_ids):
    """Per-site z-score with mean/std from TRAIN subjects of that site only
    (label-free, fold-fitted) - removes scanner offsets across 20 sites."""
    site = df["SITE_ID"].reindex(roi.index)
    tr = roi.loc[roi.index.intersection(train_ids)]
    mu = tr.groupby(site.loc[tr.index]).mean()
    sd = tr.groupby(site.loc[tr.index]).std() + 1e-8
    out = roi.copy()
    for s in site.dropna().unique():
        if s in mu.index:
            idx = site.index[site == s]
            out.loc[idx] = (roi.loc[idx] - mu.loc[s]) / sd.loc[s]
    return out


def fit_predict(Xtr, ytr, Xte):
    # train-only imputation + standardisation
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    # L2 strength chosen by inner CV on the training folds only
    inner = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
    cv = LogisticRegressionCV(Cs=np.logspace(-4, 1, 11), cv=inner,
                              scoring="roc_auc", max_iter=5000).fit(Xtr, ytr)
    clf = LogisticRegression(C=cv.C_[0], max_iter=5000)
    # out-of-fold train probs -> unbiased threshold tuning
    ptr = cross_val_predict(clf, Xtr, ytr, cv=inner,
                            method="predict_proba")[:, 1]
    return ptr, clf.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def metrics(y, p, thr):
    pred = (p > thr).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "auc": float(roc_auc_score(y, p)), "threshold": thr}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", help="step3 results dir(s)")
    ap.add_argument("--pheno", default=config.PHENO_CSV)
    ap.add_argument("--no-site", action="store_true",
                    help="drop site one-hot (stricter, site-agnostic model)")
    ap.add_argument("--no-fiq", action="store_true",
                    help="drop full-scale IQ")
    ap.add_argument("--thr-metric", choices=["bacc", "f1"], default="bacc",
                    help="criterion for the train-tuned decision threshold")
    ap.add_argument("--roi", nargs="?", const=config.FS_FEATURES, default=None,
                    help="add FreeSurfer region features (step4a csv)")
    ap.add_argument("--harmonise", action="store_true",
                    help="per-site z-score region features (train-fitted); "
                         "did not help on ABIDE I: AUC 0.759 vs 0.782")
    ap.add_argument("--out", default="results_fused.json")
    args = ap.parse_args(argv)

    df = load_pheno(args.pheno)
    roi = None
    if args.roi:
        if not os.path.exists(args.roi):
            sys.exit(f"missing {args.roi} - run step4a_fetch_freesurfer.py")
        roi = pd.read_csv(args.roi, index_col="SUB_ID")
        roi = icv_normalise(roi.loc[:, roi.notna().mean() > 0.5])
        print(f"roi: {roi.shape[1]} features, "
              f"{int(roi.notna().any(axis=1).sum())} subjects")
    cnn = load_cnn(args.dirs)
    all_ids = [i for f in cnn.values() for i in f]
    missing = [i for i in all_ids if i not in df.index]
    if missing:
        print(f"warning: {len(missing)} subjects lack phenotype rows - dropped")
    sites = sorted(df.loc[[i for i in all_ids if i in df.index], "SITE_ID"].unique())

    modes = {"cnn_only": (True, False, False), "pheno_only": (False, True, False),
             "fused": (True, True, False)}
    if roi is not None:
        modes.update({"roi_only": (False, False, True),
                      "roi+pheno": (False, True, True),
                      "cnn+roi+pheno": (True, True, True)})
    res = {m: {} for m in modes}
    for fk in cnn:
        te = [i for i in cnn[fk] if i in df.index]
        tr = [i for g, f in cnn.items() if g != fk for i in f if i in df.index]
        p = {i: v[1] for f in cnn.values() for i, v in f.items()}
        ytr = np.array([cnn[g][i][0] for g in cnn if g != fk
                        for i in cnn[g] if i in df.index])
        yte = np.array([cnn[fk][i][0] for i in te])
        roi_k = None
        if roi is not None:
            roi_k = site_harmonise(roi, df, tr) if args.harmonise else roi
        for m, (uc, up, ur) in modes.items():
            fx = lambda ids: features(df, ids, np.array([p[i] for i in ids]),
                                      uc, up, not args.no_site,
                                      not args.no_fiq, sites,
                                      roi_k if ur else None)
            ptr, pte = fit_predict(fx(tr), ytr, fx(te))
            res[m][fk] = metrics(yte, pte,
                                 best_threshold(ytr, ptr, args.thr_metric))

    out = {"runs": [os.path.abspath(d) for d in args.dirs],
           "site": not args.no_site, "fiq": not args.no_fiq}
    print(f"\nruns={len(args.dirs)} site={not args.no_site} "
          f"fiq={not args.no_fiq}  (subject-level, 5-fold)")
    print(f"{'':14s}" + "".join(f"{k:>16s}" for k in
                                ["accuracy", "f1", "auc"]))
    for m in modes:
        ov = {k: (float(np.mean([f[k] for f in res[m].values()])),
                  float(np.std([f[k] for f in res[m].values()])))
              for k in ["accuracy", "precision", "recall", "f1", "auc"]}
        out[m] = {"fold_metrics": res[m], "overall_metrics":
                  {k: {"mean": a, "std": b} for k, (a, b) in ov.items()}}
        print(f"{m:14s}" + "".join(f"{ov[k][0]:9.3f} +/-{ov[k][1]:.3f}"
                                   for k in ["accuracy", "f1", "auc"]))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=4)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
