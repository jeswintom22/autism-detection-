# Autism Detection from Structural MRI — Technical Report

**Pipeline:** staged, leakage-resistant ASD classification on ABIDE I sMRI
**Best honest result:** **72.4% ± 3.1 accuracy, 0.782 AUC** (subject-level 5-fold CV)
**Key finding:** replicating a reference method reproduced its ~96.5% "accuracy" —
and showed it comes entirely from slice-level data leakage; the same model under
subject-level evaluation scores 62.5%.

---

## 1. Objective

Build a reproducible end-to-end pipeline that detects Autism Spectrum Disorder
from structural MRI, replicating the staged workflow of a reference
disaster-image project and a published EfficientNetB5 ASD study — while
evaluating honestly (subject-level, no leakage). Deliverables: data
validation, preprocessing (skull-stripping + filtering), model
training/evaluation, orchestration, single-scan inference, and a quantified
analysis of where high published accuracy figures come from.

## 2. Dataset

- **Source:** ABIDE I (Preprocessed Connectomes style), subject IDs 50002–51607.
- **Size:** 1004 subjects — 504 control, 500 ASD (`control/`, `treatment/`).
- **Scanner sites:** 20, all in native space — at least 5 distinct volume
  shapes ((128,256,256)…(160,480,512)) and voxel sizes (1.0–1.33 mm; some
  0.5 mm in-plane). All volumes are LAS-oriented.
- **Skull state:** mostly pre-stripped — the pipeline's auto-detector
  classified 995 as stripped, 1 as needing a morphology fallback strip.
- **Phenotype:** joined on SUB_ID against the public
  `Phenotypic_V1_0b_preprocessed1.csv`; all 1004 IDs matched and the CSV
  diagnoses agreed with the folder labels for **1004/1004** subjects.

## 3. Pipeline architecture

| Stage | File | What it does |
|---|---|---|
| 1 | `step1_collect_data.py` | scans both class folders, validates NIfTI readability, writes manifest + summary |
| 2 | `step2_preprocess.py` | per-volume deskull check (nonzero-fraction + border-rim heuristics); strips only if needed; applies the paper's nimble/Canny filter; writes `preprocessed_dataset/` |
| 3 | `step3_classify.py` | slice extraction → cache → grouped-CV training (ResNet101/ResNet50/EfficientNetB5) → subject-level metrics |
| — | `ensemble_eval.py` | averages held-out probs across independent runs (the three views) |
| 4a | `step4a_fetch_freesurfer.py` | downloads ABIDE FreeSurfer 5.1 stats → 259-region feature table |
| 4 | `step4_fuse.py` | per-fold stacker: CNN prob + phenotype (+ROI) → final model |
| — | `run_all.py`, `demo.py` | orchestration; single-scan inference |

### 3.1 Input representation (2.5D slicing)

Each volume → N=9 slices along the chosen anatomical axis, spread over the
middle third (`--slice-select area` picks the N slices with the most brain
coverage instead). Each slice: min-max normalise → resize to 224² (or 456² for
B5) → grayscale ×3 channels → uint8, cached in `slice_cache/` keyed on
file+mtime+size+view+size. A subject's probability is the mean of its slice
probabilities (+ horizontal-flip TTA).

### 3.2 Leakage prevention (the load-bearing detail)

- **StratifiedGroupKFold** — every fold holds *all* slices of a subject in
  exactly one partition; a runtime assertion verifies zero subject overlap.
- **Threshold tuning** uses *training subjects only* (balanced accuracy).
- **Stacker features** are always *out-of-fold* probabilities; imputation,
  standardisation and ICV/site normalisation are fit on training subjects only.
- **Regularisation** inside the stacker is selected by a nested inner 5-fold
  CV on the training folds.
- Site harmonisation was tested (per-site z-score, train-fitted) — it
  *reduced* performance and is left off by default.

### 3.3 Models

- **CNN:** ImageNet-pretrained ResNet101 (also ResNet50, EfficientNetB5);
  only the last block + a 512→256 SiLU/BatchNorm/Dropout head are trainable;
  backbone BatchNorms frozen; AdamW, class-balanced BCE (pos_weight),
  label smoothing 0.1, discriminative LR (layer4 at 0.1×), AMP, early
  stopping on validation loss.
- **Tri-planar ensemble:** three *separate* axial/coronal/sagittal models —
  the literature's correct multi-plane design (the alternative of stacking
  the three planes as RGB channels was tested and underperformed: 0.663 acc
  vs 0.684 axial-only — the planes are not spatially co-registered).
- **Fusion stacker:** per-fold logistic regression on `[logit(p_CNN), age,
  log(age), sex, FIQ, FIQ-missing]` + optionally 259 ICV-normalised FreeSurfer
  ROI features.

## 4. Results

### 4.1 Single-view CNN runs (subject-level, 9 slices)

| View | Accuracy | AUC |
|---|---|---|
| Axial | 0.641 ± 0.046 | 0.746 |
| Coronal | 0.648 ± 0.025 | 0.717 |
| Sagittal | 0.650 ± 0.043 | 0.712 |
| **3-view ensemble** | 0.660 ± 0.033 | 0.749 |

### 4.2 Fusion ablation (step4_fuse.py, 3 views, no-site)

| Features | Accuracy | AUC |
|---|---|---|
| CNN only (retuned threshold) | 0.675 | 0.749 |
| Phenotype only (age/sex/FIQ) | 0.585 | 0.598 |
| ROI only (259 FS features) | 0.584 | 0.625 |
| ROI + phenotype | 0.604 | 0.650 |
| CNN + phenotype | 0.712 | 0.767 |
| **CNN + ROI + phenotype** | **0.724 ± 0.031** | **0.782 ± 0.025** |

Notable: the decision-threshold criterion mattered — F1-max thresholds biased
predictions toward ASD (accuracy 0.660→0.698 for the same probabilities after
switching to balanced accuracy).

### 4.3 EfficientNetB5 — the leakage experiment

The reference method was replicated faithfully: B5 @ 456², 3 mid axial
slices, `StratifiedKFold` over *slices*, fixed 0.5 threshold, no
augmentation, uniform LR — `--split slice` in `step3_classify.py`.

| Evaluation | Accuracy | AUC |
|---|---|---|
| **Slice-level (paper's protocol)** | **0.965 ± 0.009** | 0.993 |
| **Same model, subject-level (ours)** | **0.625 ± 0.024** | 0.689 |

**34-point inflation from the split alone** — matching the leakage literature's
quantified estimate (+29–55 pts). Adjacent slices of the same brain in train
and validation let the model memorise subjects instead of learning autism.

### 4.4 Multi-backbone ensemble

| Model | Accuracy | AUC |
|---|---|---|
| 4-run CNN ensemble (3 ResNet views + B5) | 0.667 | 0.752 |
| Fused incl. B5 probs | 0.704 | 0.781 |

B5 honestly underperforms ResNet101 and drags the ensemble down — kept in the
table for completeness; the headline stays the 3-view fusion.

## 5. Analysis

1. **The model ceiling is data-bound, not architecture-bound.** Three
   backbones, three views, slice counts 3→9, augmentation, TTA and
   regularisation all land in 0.64–0.70 — multisite native-space sMRI simply
   has limited class signal.
2. **Non-image features beat more CNN complexity.** Adding age/sex/FIQ
   (+FIQ most of it) lifted accuracy ~4 points; FreeSurfer morphometry adds
   ~1 more. This matches the strongest honest ABIDE results.
3. **Published ~90–95% claims on ABIDE are reproducible — by leaking.** We
   reproduced the number exactly (0.965) and quantified its source. The
   corrected literature band (65–72% subject-level, AUC ≈0.62–0.80) is where
   this pipeline sits.
4. **Scanner/site effects are real.** ROI features alone reach only 0.62
   AUC, and train-fitted per-site harmonisation made fusion *worse* —
   consistent with ASD-vs-site being weakly confounded here (site ASD ratios
   0.39–0.58).

## 6. Limitations (honest)

- Checkpoint selection per fold uses that fold's validation loss (mild
  optimism; fixing needs an inner-validation retrain).
- ROI features exist for 942/1004 subjects (62 imputed + flag).
- The fused model requires phenotype inputs at inference — for a scan-only
  deployment the CNN ensemble (0.675) is the deliverable.
- Threshold-tuned accuracy is threshold-dependent; AUC is the stabler figure.
- No fMRI/connectivity features — the next modality jump, not implemented.

## 7. Reproduction

```powershell
# data + preprocessing
python step1_collect_data.py
python step2_preprocess.py --jobs -1

# three honest view models (~15-40 min each on RTX 3050)
python step3_classify.py --n-slices 9 --view axial    --models saved_models_ax --results results_ax
python step3_classify.py --n-slices 9 --view coronal  --models saved_models_co --results results_co
python step3_classify.py --n-slices 9 --view sagittal --models saved_models_sa --results results_sa

# CNN ensemble + fused model (the 0.724/0.782 headline)
python ensemble_eval.py results_ax results_co results_sa --out results_ensemble.json
python step4a_fetch_freesurfer.py
python step4_fuse.py results_ax results_co results_sa --roi --no-site --out results_fused_roi.json

# paper-replica (labelled slice-level; reproduces ~96.5%)
python step3_classify.py --backbone efficientnet_b5 --img-size 456 --n-slices 3 --split slice --models saved_models_b5 --results results_b5
# same model honestly:
python step3_classify.py --backbone efficientnet_b5 --img-size 456 --n-slices 3 --split group --workers 0 --models saved_models_b5h --results results_b5h

# single-scan inference
python demo.py --model "saved_models_ax,saved_models_co,saved_models_sa" --ensemble scan.nii.gz
```

## 8. Artifacts

- `results_*/all_metrics.json` + `val_probs.json` — per-fold metrics and
  held-out probabilities (`"split"` field marks slice-level runs).
- `results_*/training_metrics/`, `confusion_matrices/` — curves and matrices.
- `results_fused_roi.json`, `results_ensemble.json`, `results_fused_b5.json`,
  `results_ens4.json` — fusion/ensemble outputs.
- `saved_models_*/model_fold_*.pt` — checkpoints with stored config
  (view, slices, img_size, threshold, TTA, split).
- `context.md` — detailed engineering log; `README.md` — usage.

## 9. Conclusion

An end-to-end, reproducible ASD MRI pipeline reaches **72.4% ± 3.1 accuracy /
0.782 AUC** under strict subject-level evaluation — at the top of the honest
multisite-ABIDE band for structural MRI. Replicating a reference ~95%-claim
method showed its number is produced entirely by slice-level leakage
(0.965 → 0.625 under the corrected protocol) — the pipeline's grouped-CV
design is the reason its own numbers are trustworthy.
