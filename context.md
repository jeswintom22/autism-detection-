# Project Context — ASD Detection from sMRI

Handoff document for anyone (human or agent) continuing this work. Read this
first, then `README.md` for usage. Last updated: 2026-11-01.

---

## 1. Goal

Classify **Autism Spectrum Disorder (ASD) vs. typically-developing control**
from skull-stripped structural T1 MRI (ABIDE-style), using:

1. adaptive **"Canny-nimble" filter** preprocessing, then
2. **ResNet101** transfer learning on axial slices,
3. evaluated with **subject-level 5-fold cross-validation**.

This follows the team's paper draft (`C:\Users\Jeswin\Downloads\Autism Detection.pdf`).
The paper reports ResNet-101 0.901, VGG19 0.904, Inception-v3 0.932, and ~95%
average accuracy for the proposed method, with methods [12, 14, 16] used as comparisons.

The repo layout copies the staged structure of an earlier disaster-image
project the user built (step1 → step2 → step3 → run_all → demo). Per the
user's choice, the **Flask API, React frontend, and MLflow parts were left out**
(scope = pipeline only).

---

## 2. Locations

| What | Path |
|---|---|
| Repo (git, remote `github.com/jeswintom22/autism-detection-`) | `D:\study\autisim\MINI PROJECT\autism-detection-\` |
| Dataset — control (label 0) | `D:\study\autisim\control\` — 504 × `<id>_skull_stripped.nii.gz` |
| Dataset — ASD (label 1) | `D:\study\autisim\treatment\` — 500 × `<id>_skull_stripped.nii.gz` |
| Original reference scripts (not in repo) | `C:\Users\Jeswin\Downloads\asd_resnet_pipeline.py` (TF/Keras, Colab), `canny_nimble.py` |
| Paper draft | `C:\Users\Jeswin\Downloads\Autism Detection.pdf` |
| Unrelated sibling project (do not touch) | `D:\study\autisim\Quantum-ML-for-Disease-Classification-\` |

Dataset facts (verified): the scans are flat folders with no subdirectories.
Every scan is 176×256×256 (the 3rd axis, with 256 slices, is the one we slice).
About 17–30% of voxels are nonzero, and every scan is already skull-stripped.
Total size is about 2 GB. The dataset is **outside** the repo; `config.DATA_ROOT`
defaults to `..\..\` from the repo, i.e. `D:\study\autisim`.

---

## 3. Environment

- Windows, Python **3.10.11** (`C:\Users\Jeswin\AppData\Local\Programs\Python\Python310`)
- GPU: **NVIDIA RTX 3050 6 GB Laptop**, CUDA works
- Installed: `torch 2.5.1+cu121`, `torchvision 0.20.1`, nibabel 5.4, opencv 5.0,
  scikit-learn 1.7, scipy 1.15, matplotlib 3.10, joblib 1.5, numpy 2.2
- **Not installed:** tensorflow (the training script was ported to PyTorch for this
  reason; TF has no GPU support on native Windows), seaborn (plain matplotlib is
  used instead), HD-BET (optional)
- ImageNet ResNet101 weights are cached at `C:\Users\Jeswin\.cache\torch\hub\checkpoints\`
- Shell gotcha: in Git Bash, `$TEMP` is `/tmp`. MSYS converts it to a Windows path
  when it is passed to native python.

---

## 4. Pipeline (repo files)

```
control/ + treatment/
  -> step1_collect_data.py   manifest.csv, step1_summary.json
  -> step2_preprocess.py     preprocessed_dataset/<class>/<id>_filtered.nii.gz, step2_summary.json
  -> [run_all.py validation gate]
  -> step3_classify.py       saved_models/*.pt, results/
  -> demo.py                 P(ASD) for one scan
```

| File | What it does |
|---|---|
| `config.py` | Every path and hyper-parameter. Paths can be overridden with `ASD_DATA_ROOT`, `ASD_CONTROL_DIR`, `ASD_TREATMENT_DIR`, `ASD_PREPROCESSED_DIR`. |
| `step1_collect_data.py` | Walks both class folders. Checks that each header and middle slice can be read. Writes `manifest.csv` with columns subject_id, label, class_name, path, shape, n_voxels, nonzero_frac, status, error. |
| `step2_preprocess.py` | Runs per subject in parallel with joblib, and skips outputs that already exist so it can resume. **Deskull:** `is_stripped()` checks two things: volume nonzero fraction ≤ 0.45, and nonzero fraction in a 4-voxel border slab ≤ `STRIP_MAX_RIM`. If a scan looks un-stripped, it runs HD-BET when installed and otherwise falls back to morphology (35th-percentile threshold → opening → largest connected component → fill holes → closing → erosion). **Nimble filter** (vectorised, per axial slice): `K = g·f + (1−g)·M + T·(f−M)`, where g = 0.4, M = 4-neighbour mean (paper eq. 2), and T = local Canny edge density × (1 − local noise std / 255). Output is float32 in the 0–255 range, with affine and header kept. |
| `step3_classify.py` | PyTorch port of `asd_resnet_pipeline.py`; details in §5. |
| `run_all.py` | Runs step1 → step2 → gate (both class folders exist and are non-empty; warns about step2 failures) → step3. Flags: `--skip-collect`, `--skip-preprocess`, `--max`, `--limit`, `--folds`, `--epochs`, `--batch`, `--raw`, `--backbone`. |
| `demo.py` | Loads `best_model.pt` with `weights_only=True`. Raw input is deskull-checked and nimble-filtered in memory; `*_filtered*` files are used as they are. Reads `n_slices`, `planes`, `tta`, and `threshold` from the checkpoint. `--ensemble` averages all `model_fold_*.pt`. |
| `requirements.txt`, `.gitignore` | Generated data, models, results, and the manifest/summary files are git-ignored. |

---

## 5. Training details (`step3_classify.py`)

- **Input:** for each subject, take N axial slices. Each slice is min-max scaled,
  resized to 224×224 with `INTER_AREA`, repeated to 3 channels, and stored as uint8.
  The whole dataset is held in RAM.
  - `--n-slices ≤ 3`: mid-1, mid, mid+1 (paper / original-script behaviour, the default)
  - `--n-slices > 3`: evenly spread across the middle third of the volume
    (with 9 slices on 256 slices: indices 86…170)
- **Normalisation:** ImageNet mean/std, applied inside the Dataset.
- **Augmentation** (train only, on by default; turn off with `--no-augment`):
  left-right flip (p = 0.5); affine with ±8° rotation, 0.95–1.05 scale, and ±5% shift (p = 0.7);
  intensity ×0.9–1.1 and ±12 (p = 0.5). Each fold uses its own seed.
- **Model:** torchvision `resnet101` with `IMAGENET1K_V2` weights (`--backbone resnet50` is also supported).
  Everything is frozen except `layer4`. **Every backbone BatchNorm2d stays frozen:** it is set to eval
  and re-frozen after each `model.train()`. The head is
  `Linear(2048,512)-SiLU-BN-Drop(.5)-Linear(512,256)-SiLU-BN-Drop(.3)-Linear(256,1)`.
- **Optimisation:** AdamW with **discriminative LR** — layer4 at `--lr × --bb-lr-mult`
  (default 1e-5) and head at `--lr` (1e-4), wd 1e-5; **label smoothing** 0.1 on train
  targets (`--smooth`); `BCEWithLogitsLoss(pos_weight = neg/pos)`;
  ReduceLROnPlateau (×0.2, patience 5, min 1e-6); early stopping on val loss
  (patience 10), which restores the best weights (held on CPU); AMP + channels_last +
  cudnn benchmark on CUDA; batch 16.
- **Multi-plane input (`--planes 3`):** each image's 3 channels = axial/coronal/sagittal
  slices at matching spread positions (2.5-D input). Default `--planes 1` = axial only.
- **Eval extras:** flip **TTA** on all eval predictions (`--no-tta` disables); per-fold
  **decision threshold tuned on the training subjects only** (max F1 over 181 points —
  no val peeking), stored in `fold_metrics` and each checkpoint, used by `demo.py`.
- **CV:** `StratifiedGroupKFold(5, shuffle, seed 42)`, grouped by subject. An assert checks
  that no subject appears in both train and val. `--folds` must be ≥ 2.
- **Slice cache:** extracted slices saved as `.npy` under `slice_cache/` (`--cache-dir`,
  keyed by abspath + n_slices + planes; ~200 s → <1 s reload). `''` disables.
- **Metrics:** computed **per subject** by averaging slice probabilities at the tuned
  threshold: accuracy, precision, recall, F1, AUC, and confusion matrix.
  `best_model.pt` = the fold with the highest F1.
- **Outputs:** `saved_models/model_fold_k.pt` and `best_model.pt`. Each checkpoint is a dict with
  state_dict, backbone, img_size, n_slices, fold, and metrics.
  `results/all_metrics.json` holds per-fold metrics plus overall mean/std.
  `results/training_metrics/fold_k_{loss,acc}.png` and
  `results/confusion_matrices/*.png` hold the plots.
- **Smoke test** (sends output to a temp folder so real artifacts are not overwritten):
  `python step3_classify.py --limit 8 --folds 2 --epochs 1 --batch 4 --models <tmp>/m --results <tmp>/r`

---

## 6. History — what was done, in order

1. **Built the repo:** config, step1/2/3, run_all, demo, README, requirements, .gitignore.
2. **Deskull threshold fix:** the first rim threshold (0.02) marked real stripped
   scans as un-stripped, because ABIDE stripped scans still have 1–7% nonzero rim voxels.
   It was raised to `STRIP_MAX_RIM = 0.15`.
3. **Bugs fixed while porting `canny_nimble.py`:**
   - `adaptive_window_sizes` was computed but never used.
   - `addWeighted(0.4, 0.3)` darkened images because the weights only add up to 0.7.
   - A Python loop over each window was replaced with box filters, which is much faster.

   These were replaced with a direct implementation of paper eq. 1.
4. **Other fixes:** the confusion-matrix plot crashed because `fmt='d'` was used on floats.
   The `±` character was replaced with `+/-` because the Windows console could not print it.
   `--folds ≥ 2` is now enforced.
5. **Full baseline run** by the user: `python run_all.py`, with 3 slices and no augmentation.
   - step1: 1004/1004 scans OK.
   - step2: 3.7 minutes, 0 failures. 995 scans were already stripped, 1 was stripped with the morphology
     fallback, and 8 were cached. (Which subject needed the morphology fallback has not been checked yet.)
   - step3: loading took 201 s. Each fold early-stopped at epochs 11–13.
6. **Added augmentation and `--n-slices`** (the spread option), and `demo.py` now reads `n_slices` from the checkpoint.
7. **Incident:** an agent smoke test was written to the real `saved_models/` and `results/`
   folders and overwrote the baseline fold 1/2 checkpoints, their plots, `best_model.pt`, and
   `all_metrics.json`. Recovery:
   - `all_metrics.json` was rebuilt from the console output.
   - `best_model.pt` was restored by copying baseline `model_fold_3.pt` (the best F1 fold).
   - The contaminated files were deleted.

   **Rule from now on:** smoke tests must use `--models`/`--results` pointing at temp folders.
8. **Code-review fixes:**
   - The README smoke example now uses `--folds 2`.
   - `get_fdata(dtype=np.float32)` is used in step2, step3, and demo.
   - The flipped image is made contiguous before `cv2.warpAffine`.
   - `slice_indices` is clamped so it stays inside the volume.
   - Step3 exits clearly when no scans load.
   - The best-epoch weights are kept on CPU.
   - The unused `copy` import was removed.

   All of these passed a smoke test into a temp folder.
9. **The user ran the 9-slice + augmentation run** (started 20:33, finished ~21:48):
   `python step3_classify.py --n-slices 9 --folds 5 --epochs 30`. It used the code from
   **before** step 8; none of the step-8 fixes change results. Improvement across all
   metrics — see §7.
10. **Accuracy + speed bundle** (step3 + demo + config):
    - per-fold threshold tuned on train subjects only (saved in checkpoints/metrics, used by demo)
    - flip TTA on eval predictions (`--no-tta`)
    - discriminative LR: layer4 `--bb-lr-mult` 0.1 × head LR
    - label smoothing `--smooth` 0.1
    - `--planes 3` multi-plane input (axial/coronal/sagittal channels)
    - `demo.py --ensemble` (mean prob over all fold checkpoints, ckpt threshold + TTA)
    - slice `.npy` cache (`--cache-dir`, abspath+n_slices+planes key; cold ~28 s/16 subj → warm 0.2 s)
    - channels_last + cudnn.benchmark; cache key uses abspath (relative dirs no longer bust it)
    Verified with temp-dir smoke (planes 3, cache hit, ensemble demo).

---

## 7. Results so far

### Baseline — 3 slices, no augmentation, nimble-filtered input (subject-level, 5-fold)

| Fold | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| 1 | 0.592 | 0.846 | 0.220 | 0.349 | 0.594 |
| 2 | 0.627 | 0.651 | 0.540 | 0.590 | 0.696 |
| 3 | 0.667 | 0.714 | 0.550 | 0.622 | 0.695 |
| 4 | 0.590 | 0.616 | 0.455 | 0.523 | 0.653 |
| 5 | 0.617 | 0.700 | 0.416 | 0.522 | 0.687 |
| **Mean ± std** | **0.618 ± 0.028** | 0.705 ± 0.079 | 0.436 ± 0.119 | **0.521 ± 0.094** | **0.665 ± 0.039** |

What these numbers say:
- The model **overfits badly**. Train accuracy reaches about 100% by epoch around 11,
  while val accuracy stays at 0.58–0.67 and val loss rises from epoch 2–3.
- Precision is much higher than recall, so the model under-calls ASD at the 0.5 threshold.
  Threshold tuning or calibration is an open item (see §9).
- About 62–67% is in line with honest **subject-level** CV on ABIDE sMRI. The ~95% in the paper
  probably comes from a different setup:
  - The original Colab script read from `pre_seg_final`, which is probably segmented GM/WM maps.
  - The paper may have split by slice rather than by subject, or have site confounds.

  Check how the paper's protocol was set up before comparing the numbers directly.

### 9-slice + augmentation run — COMPLETE (subject-level, 5-fold)

`python step3_classify.py --n-slices 9 --folds 5 --epochs 30`, finished ~21:48 on 2026-11-01.
All `saved_models/` checkpoints, `best_model.pt` (fold 3), and `results/` now belong to this run.

| Fold | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| 1 | 0.692 | 0.852 | 0.460 | 0.597 | 0.760 |
| 2 | 0.592 | 0.583 | 0.630 | 0.606 | 0.694 |
| 3 | 0.746 | 0.758 | 0.720 | 0.739 | 0.813 |
| 4 | 0.720 | 0.853 | 0.525 | 0.650 | 0.735 |
| 5 | 0.672 | 0.727 | 0.555 | 0.629 | 0.729 |
| **Mean ± std** | **0.684 ± 0.053** | 0.755 ± 0.099 | 0.578 ± 0.090 | **0.644 ± 0.051** | **0.746 ± 0.039** |

vs. baseline (3 slices, no augmentation): acc +6.6 pts, F1 +12.3, AUC +8.1.
- Overfitting is reduced, not gone: train acc peaked ~0.86 (was ~1.00), val loss no longer
  explodes past ~1.2, early stop later (epochs 14–20 vs 11–13).
- Precision still exceeds recall (0.755 vs 0.578) — threshold tuning is still the
  cheapest remaining gain.
- One anomaly: fold 5 val_loss spiked to 12.9 at epoch 6 then recovered — transient
  instability, probably an unlucky augmented batch; no action needed unless it recurs.

### planes=3 channel-stacked run — COMPLETE (2026-11-01 evening)

`python step3_classify.py --n-slices 9 --planes 3 --folds 5 --epochs 50`
(aug + TTA + threshold tuning + label smoothing + discriminative LR all on;
parallel slice loading wrote `slice_cache/`, load ~121 s for 1004 subjects)

| | acc | prec | rec | f1 | auc |
|---|---|---|---|---|---|
| Mean ± std | 0.663 ± 0.019 | 0.654 ± 0.038 | 0.702 ± 0.054 | 0.675 ± 0.013 | 0.741 ± 0.023 |

- **Below the axial-only 9-slice run** (0.684 acc / 0.746 AUC) on acc/AUC; F1/recall
  higher only because tuned thresholds dropped (0.44–0.51 vs 0.5). Interpretation:
  the 3 channels are independently-spread, NON co-registered cross-sections —
  ImageNet pretrained convs expect aligned channels. This is why literature
  multi-plane methods fuse LATE (per-plane models + voting), not stack channels.
- Artifacts for this run currently sit in `saved_models/` + `results/`; the
  axial 9-slice best run is backed up in `runs/n9_aug/`.

### New flags added after the planes=3 result (2026-11-01)

- `--view axial|coronal|sagittal` — single-view training for `planes=1`
  (axis stored in checkpoint as `"axis"`; demo uses it).
- `--slice-select spread|area` — `area` picks the top-coverage (nonzero-fraction)
  slices within the middle half (anatomical slice selection).
- `run_cv` now writes `results/val_probs.json` — per-fold held-out
  basename/label/prob + threshold, for cross-run ensembling.
- `ensemble_eval.py <results_dir1> <results_dir2> ...` — joins on basename,
  averages probs, recomputes honest metrics (folds identical across runs
  because the split seed is fixed).
- `demo.py --ensemble` now accepts `--model` = dir(s) and/or file(s),
  comma-separated; per-config input arrays are cached so mixed-view ensembles work.
- `--jobs -1` parallel slice loading is the default (`0` = sequential);
  cache key includes mtime+size; writes are atomic.
- `--workers N` (default 4) — DataLoader workers on the TRAIN loader only
  (eval loaders stay at 0). X is stored as a shared-memory torch tensor so
  Windows-spawned workers don't each copy 1-4 GB; augmentation seed is
  deterministic per (epoch, index) so worker order doesn't matter. TF32
  enabled on CUDA.
- Verified: coronal+area smoke (2.8 s load/16 subj), 2-view ensemble_eval,
  4-checkpoint cross-view demo.

## Full 3-view runs + phenotype fusion
- Dataset = ABIDE I (IDs 50002-51607, 20 sites, native space, mixed
  shapes/voxel sizes). All 1004 IDs match the ABIDE phenotypic CSV and
  labels agree 100%.
- Views (F1-threshold): axial 0.641/0.746 AUC, coronal 0.648/0.717,
  sagittal 0.650/0.712, 3-view ensemble 0.660/0.749.
- Threshold criterion switched to balanced accuracy (`--thr-metric`,
  default `bacc`); max-F1 was biasing toward predicting ASD.
- `step4_fuse.py`: per-fold logistic stacker on out-of-fold CNN prob +
  age/sex/FIQ(/site); impute/scale/threshold fit on training folds only.
  3-view: cnn_only 0.698/0.749, fused 0.718/0.764, fused w/o site
  0.713/0.767, w/o site+FIQ 0.681/0.750. FIQ carries the gain.
- EARLY_STOP_PATIENCE 10 -> 5 (best val loss always at epoch 1-5).

## FreeSurfer region features (step4a + `step4_fuse.py --roi`)
- ABIDE Preprocessed FreeSurfer 5.1 stats (aseg + lh/rh aparc) from the
  fcp-indi bucket via FILE_ID; 942/1004 subjects have them (62 failed PCP QC
  -> imputed + missing flag). 259 features; volumes/areas divided by ICV.
- Stacker now picks L2 strength with inner 5-fold CV on training folds and
  tunes the threshold on inner out-of-fold probs (no in-sample bias).
- Results (3-view, no site, with FIQ): fused cnn+pheno 0.712/0.767,
  roi_only 0.584/0.625, **cnn+roi+pheno 0.724/0.782** (best).
- Per-site z-score harmonisation (train-fitted) hurt: 0.694/0.759 -> opt-in
  `--harmonise`.

## EfficientNetB5 paper replica (branch effnet-b5-replica)
- `--backbone efficientnet_b5` + `--img-size` + `--split {group,slice}` in
  step3. slice mode = faithful replica of the reference code: slice-level
  StratifiedKFold (leak: same subject in train+val), thr 0.5, no aug/TTA,
  uniform LR, patience 10, slice-level metrics. Outputs tagged
  `"split": "slice"`; banner warns; subject-level metrics still recorded.
- B5 head identical to resnet head (512->256, SiLU/BN/Dropout); trainable =
  last two features blocks (~last 30 Keras layers); backbone BNs frozen.
- Cache tag gains `i<img_size>` when !=224; checkpoints store img_size and
  split; demo reads img_size from checkpoint.
- ensemble_eval/step4_fuse warn when fed slice-level val_probs.
- Smoke (16 subj, 2 ep): slice mode ran, AUC 0.896 already inflated;
  group mode unchanged.

## B5 results (real run)
- `--split slice` replica: **0.965 +/- 0.009 acc / 0.993 AUC** slice-level —
  reproduces the paper's ~95% claim; driven entirely by same-subject slices
  in train+val (saved_models_b5, results_b5).
- `--split group` honest B5: **0.625 +/- 0.024 acc / 0.689 AUC**
  (saved_models_b5h, results_b5h, log in results_b5h_log.txt) — B5 is the
  weakest backbone tried; the ~90% was pure leakage.
- Known issue: DataLoader `--workers >0` hangs on Windows spawn for 456²
  (~1.5 GB tensor shared-mem copy stalls) — honest run used `--workers 0`.
- 4-run CNN ensemble (+B5): 0.667/0.752 (vs 0.660/0.749 for 3 views).
- Fused cnn+roi+pheno with B5: 0.704/0.781 — does NOT beat 3-view fused
  (0.724/0.782). **Best honest result stays: 3 ResNet views + ROI + pheno.**

---

## 8. Conventions / rules for continuing

- **Never point a smoke or test run at the real `saved_models/` or `results/`.**
  Always pass `--models` and `--results` pointing at a temp folder.
- Before using the GPU, check whether a training job is already running, because only 6 GB of VRAM is available:
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` and look for `step3`.
- **Folds must stay split by subject (`StratifiedGroupKFold`) and metrics must stay per subject.**
  Do not switch to slice-level splits; that leaks data between train and val and inflates accuracy.
- Keep all knobs in `config.py` or CLI flags. Do not hard-code paths.
- Do not modify the dataset folders. Preprocessing only writes to `preprocessed_dataset/`.
- Delete `preprocessed_dataset/` only if step2 logic changes, because step2 skips outputs that already exist.
- To save a run's results before the next one overwrites them, copy `saved_models/` and `results/` to
  e.g. `runs/<name>/` (also git-ignore it). There is no automatic per-run folder yet; see §9.
- Code style: compact, no new dependencies unless they are needed, and stick to matplotlib (seaborn is not installed).
- Git: everything after the initial commit is **uncommitted** (`git status` shows the new files and the
  modified README). Nothing has been pushed. Commit only when the user asks.

---

## 9. Open items / next steps (rough priority)

1. **Run the per-view ensemble experiment** (the literature-backed multi-plane):
   ```powershell
   python step3_classify.py --n-slices 9 --view axial    --models saved_models_ax --results results_ax
   python step3_classify.py --n-slices 9 --view coronal  --models saved_models_co --results results_co
   python step3_classify.py --n-slices 9 --view sagittal --models saved_models_sa --results results_sa
   python ensemble_eval.py results_ax results_co results_sa --out results_ensemble.json
   ```
   First runs build the coronal/sagittal caches (~2 min each); compare ensemble
   vs axial-only best (0.684/0.746).
2. **`--slice-select area` run** on axial 9-slice — isolates anatomical slice
   selection vs fixed positions.
3. **Ablations for the paper:** `--no-augment` with `--n-slices 9`; `--raw`;
   `--backbone resnet50`. All reuse `slice_cache/` → ~10 s loads.
4. **CNN→feature-extractor→SVM/RF** per fold on head features (literature's
   small-data trick); and ROI/atlas features via nilearn if scans are
   MNI-registered (check one `img.affine` first).
5. **Cut overfitting further:**
   - Train only the head (freeze `layer4` too), or unfreeze only part of layer4.
   - Raise dropout and weight decay.
6. **Inception-v3 / VGG19 backbones**, to reproduce paper Table 1. Inception needs 299×299 input and has aux logits.
7. Check the one subject that went through the morphology skull-strip fallback in step2: find it with
   `preprocessed_dataset` plus a re-run of `is_stripped`, and make sure the mask looks right.
8. ~~Speed up loading~~ — done via `slice_cache/` .npy cache (~200 s → <1 s).
9. Possibly add these later (the template had them, but they were left out by choice):
   - MLflow tracking
   - a Flask `/api/predict` endpoint
   - a frontend
   - Grad-CAM saliency for interpretability

---

## 10. Quick commands

```bash
cd "D:\study\autisim\MINI PROJECT\autism-detection-"
python run_all.py                                        # full pipeline
python run_all.py --skip-collect --skip-preprocess       # retrain only
python step3_classify.py --n-slices 9                    # current best config
python step3_classify.py --n-slices 9 --no-augment       # ablation
python step3_classify.py --raw                           # ablation: unfiltered input
python demo.py D:\study\autisim\control\50030_skull_stripped.nii.gz
```
