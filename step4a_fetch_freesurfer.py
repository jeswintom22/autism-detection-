"""Step 4a - fetch ABIDE Preprocessed FreeSurfer 5.1 stats and build a
per-subject region feature table (atlas-space morphometry).

For every subject in the preprocessed dataset, downloads (once, cached)
  stats/aseg.stats       subcortical volumes + global measures (ICV, ...)
  stats/{lh,rh}.aparc.stats   Desikan-Killiany thickness / area / GM volume
from the public fcp-indi bucket and writes freesurfer_features.csv
(index SUB_ID). Subjects without a FreeSurfer run (FILE_ID 'no_filename' =
failed PCP QC) get an empty row; step4_fuse.py imputes them per fold.

    python step4a_fetch_freesurfer.py
"""
import argparse
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import config
from step4_fuse import load_pheno

FS_URL = ("https://s3.amazonaws.com/fcp-indi/data/Projects/"
          "ABIDE_Initiative/Outputs/freesurfer/5.1/{fid}/stats/{name}")
FILES = ["aseg.stats", "lh.aparc.stats", "rh.aparc.stats"]


def subject_ids(root):
    ids = []
    for cls in config.LABELS:
        for r, _, fs in os.walk(os.path.join(root, cls)):
            ids += [int(m.group(1)) for f in fs
                    if (m := re.match(r"(\d+)", f)) and f.endswith(".nii.gz")]
    return sorted(set(ids))


def fetch(fid, cache):
    """Download the 3 stats files for one FILE_ID; returns local dir or None."""
    d = os.path.join(cache, fid)
    os.makedirs(d, exist_ok=True)
    for name in FILES:
        p = os.path.join(d, name)
        if os.path.exists(p):
            continue
        for attempt in range(4):
            try:
                with urllib.request.urlopen(FS_URL.format(fid=fid, name=name),
                                            timeout=60) as r:
                    data = r.read()
                break
            except urllib.error.HTTPError:
                return None                          # not on the bucket
            except (OSError, TimeoutError) as e:     # transient network
                if attempt == 3:
                    print(f"giving up {fid}/{name}: {e}")
                    return None
                time.sleep(2 ** attempt)
        tmp = f"{p}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, p)
    return d


def _measures(text, prefix):
    out = {}
    for line in text.splitlines():
        if line.startswith("# Measure"):
            parts = [x.strip() for x in line[len("# Measure"):].split(",")]
            if len(parts) >= 4:
                try:
                    out[f"{prefix}{parts[1]}"] = float(parts[3])
                except ValueError:
                    pass
    return out


def parse(d):
    feats = {}
    with open(os.path.join(d, "aseg.stats")) as f:
        text = f.read()
    feats.update(_measures(text, "g_"))
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        c = line.split()                     # Index SegId NVoxels Vol Name ...
        if "hypointensities" not in c[4].lower():
            feats[f"vol_{c[4]}"] = float(c[3])
    for hemi in ("lh", "rh"):
        with open(os.path.join(d, f"{hemi}.aparc.stats")) as f:
            text = f.read()
        feats.update(_measures(text, f"g_{hemi}_"))
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            c = line.split()   # Name NumVert SurfArea GrayVol ThickAvg ...
            feats[f"area_{hemi}_{c[0]}"] = float(c[2])
            feats[f"gmv_{hemi}_{c[0]}"] = float(c[3])
            feats[f"thk_{hemi}_{c[0]}"] = float(c[4])
    return feats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=config.PREPROCESSED_DIR)
    ap.add_argument("--pheno", default=config.PHENO_CSV)
    ap.add_argument("--cache", default=config.FS_CACHE)
    ap.add_argument("--out", default=config.FS_FEATURES)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args(argv)

    df = load_pheno(args.pheno)
    ids = subject_ids(args.data_dir)
    fids = {i: df.loc[i, "FILE_ID"] for i in ids if i in df.index}
    todo = {i: f for i, f in fids.items() if f != "no_filename"}
    print(f"{len(ids)} subjects, {len(todo)} with a FreeSurfer FILE_ID")
    with ThreadPoolExecutor(args.threads) as ex:
        dirs = dict(zip(todo, ex.map(lambda f: fetch(f, args.cache),
                                     todo.values())))
    rows, bad = {}, 0
    for i in ids:
        d = dirs.get(i)
        if d is None:
            rows[i] = {}
            continue
        try:
            rows[i] = parse(d)
        except Exception as e:  # noqa: BLE001
            print(f"parse failed {i}: {e}")
            rows[i], bad = {}, bad + 1
    feat = pd.DataFrame.from_dict(rows, orient="index")
    feat.index.name = "SUB_ID"
    have = int(feat.notna().any(axis=1).sum())
    print(f"features: {feat.shape[1]} columns, {have}/{len(ids)} subjects "
          f"with FreeSurfer stats ({bad} parse failures)")
    feat.to_csv(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
