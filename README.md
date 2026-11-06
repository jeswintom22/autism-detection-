# Autism Spectrum Disorder Detection from Structural MRI

**EfficientNet-B5 + handcrafted texture/saliency features + Random Forest**
on the ABIDE structural MRI cohort — an end-to-end pipeline that detects
Autism Spectrum Disorder (ASD) from a subject's T1-weighted brain scan.

> **Headline result: \~76% cross-validated classification accuracy**
> (5-fold stratified CV over 18,000+ extracted slice samples, 1,003 subjects).

---

## 1. Overview

The system learns to distinguish ASD from neurotypical control brains using
two complementary views of each MRI slice:

| Branch          | What it sees                                                   | Learner                                                                 |
| --------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Deep**        | Saliency-enhanced 2-D axial slices                             | EfficientNet-B5 fine-tuned end-to-end with a custom classification head |
| **Handcrafted** | GLCM texture + LBP texture + saliency-map statistics per slice | Random Forest (500 trees)                                               |

A final ablation grid evaluates each feature family alone and in
combination, so the contribution of every component is measurable.

---

## 2. Pipeline architecture

```javascript
            T1-weighted sMRI  (control/ + treatment/, NIfTI)
                     │
                     ▼
        ┌─────────────────────────┐
        │   1 · PREPROCESSING     │
        │  canonical orientation  │
        │  resample → 1 mm³       │
        │  1–99 percentile norm   │
        │  brain crop (8 mm pad)  │
        └───────────┬─────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │   2 · NIMBLE FILTER     │   adaptive edge-preserving denoise
        │  K = g·f + (1−g)·M      │   (per axial slice, paper eq. 1–2)
        │      + T·(f − M)        │
        └───────────┬─────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │  3 · SKULL STRIP (BET)  │   FSL/HD-BET neural brain extraction
        │  already-stripped scans │   (auto-detected → pass-through)
        │  pass a rim/QC check    │
        └───────────┬─────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │   4 · AXIAL SLICING     │   9 slices @ 20–80 % occupied range
        │   + saliency enhance    │   LMS opponent-pyramid attention:
        │   (α·I + (1−α)·I ⊙ S)  │   enhanced slice → CNN input
        └───────────┬─────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────────┐   ┌──────────────────────────┐
│ 5a · DEEP BRANCH │   │ 5b · HANDCRAFTED BRANCH  │
│ EfficientNet-B5  │   │ GLCM   20-D / slice      │
│ (ImageNet init,  │   │   contrast, dissimilarity│
│  last 3 blocks   │   │   homogeneity, energy,   │
│  unfrozen)       │   │   correlation ×4 angles  │
│ head:            │   │ LBP    16-D / slice      │
│  GAP→512→256,    │   │   uniform hist + moments │
│  SiLU+BN+Dropout │   │ Saliency 6-D / slice     │
│ →logit (BCE)     │   │   map moments            │
└────────┬─────────┘   └───────────┬──────────────┘
         │                         │
         ▼                         ▼
┌──────────────────┐   ┌──────────────────────────┐
│ 6a · CNN SCORE   │   │ 6b · RANDOM FOREST       │
│ slice→P(ASD),    │   │ per-slice feature        │
│ subject vote     │   │ vectors → RF 500 trees   │
└────────┬─────────┘   └───────────┬──────────────┘
         └────────────┬────────────┘
                      ▼
        ┌─────────────────────────┐
        │ 7 · STRATIFIED 5-FOLD   │   metrics: accuracy, balanced acc,
        │   CROSS-VALIDATION      │   sensitivity, specificity, AUROC
        └─────────────────────────┘
```

---

## 3. Components

### 3.1 Improved nimble filter

Each axial slice is denoised with the adaptive nimble operator

`K = g·f + (1−g)·M + T·(f−M)`

where `M` is the 4-neighbour mean and `T` is the local Canny-edge density
damped by local noise — sharpening boundaries while smoothing homogeneous
regions (window 5, `g = 0.4`).

### 3.2 Skull stripping — BET

Brain extraction uses the BET family: FSL `bet` where available, otherwise
**HD-BET** (nnU-Net brain extraction, GPU-accelerated). Scans that already
pass the rim/nonzero-fraction QC heuristic are passed through untouched —
the method used is recorded per subject (`fsl-bet` / `hd-bet` / `morphology` /
`passed`).

### 3.3 Saliency-driven enhancement

An LMS-colour-opponent center–surround pyramid (levels 2–5) produces a
saliency map `S` per slice; the CNN sees the attention-weighted image
`I_enh = α·I + (1−α)·I ⊙ S` (α = 0.5), biasing the network toward
informative cortex regions. Six saliency-map statistics per slice
(mean, std, skew, kurtosis, energy, entropy) double as handcrafted
predictors.

### 3.4 EfficientNet-B5 classifier

ImageNet-initialised B5 (456×456 input) fine-tuned end-to-end: last three
backbone blocks unfrozen, frozen BatchNorm statistics elsewhere, custom head
`GAP → Linear(512) → SiLU → BatchNorm → Dropout(0.3) → Linear(256) → SiLU →
BatchNorm → Dropout(0.3) → logit`. Trained with
`BCEWithLogitsLoss(pos_weight)`, AdamW (discriminative LRs: backbone 1e-4 /
head 1e-3), cosine schedule, flip/±8°-rotation augmentation, bf16-AMP +
GradScaler on CUDA, early stopping on held-out slices.

### 3.5 Handcrafted features

- **GLCM (20-D/slice):** masked 8-level co-occurrence at distance 1 over
  4 angles {0°, 45°, 90°, 135°} × {contrast, dissimilarity, homogeneity,
  energy, correlation}
- **LBP (16-D/slice):** 10-bin uniform-LBP histogram (P=8, R=1) plus six
  moments over P=16, R=2 codes (mean, std, skew, kurtosis, energy, entropy)
- **Saliency (6-D/slice):** map statistics from §3.3

### 3.6 Random Forest

`n_estimators = 500`, balanced subsampling, seeds fixed. Six evaluation
families: `cnn_only` (end-to-end B5), `texture_only`, `saliency_only`,
`deep_only` (256-D slice embeddings), `mri_hybrid` (deep+texture) and
`combined` (all features). Every slice is cached per subject with a
content-hash recipe so runs are fully resumable and reproducible.

---

## 4. Evaluation protocol

- **Cohort:** 1,004 ABIDE structural scans (≈ balanced ASD / control),
  audited volume-by-volume; 1,003 usable subjects after QC.
- **Samples:** 9 axial slices per subject × near-duplicate augmentation
  → **18,054 slice samples**.
- **Validation:** stratified 5-fold cross-validation; every sample scored
  exactly once. Reported per-slice accuracy plus subject-level majority
  vote (average slice probabilities per subject → 0.5 threshold).

---

## 5. Results

### 5.1 Headline model — EfficientNet-B5 end-to-end classifier

| Metric                         | 5-fold CV |
| ------------------------------ | --------- |
| **Accuracy**                   | **0.760** |
| Balanced accuracy              | 0.758     |
| Sensitivity (ASD)              | 0.771     |
| Specificity (control)          | 0.749     |
| AUROC                          | 0.843     |
| Subject majority-vote accuracy | **0.802** |

### 5.2 Feature-family ablation (Random Forest, 500 trees)

| Family                           | Slice accuracy | Subject vote |
| -------------------------------- | -------------- | ------------ |
| cnn\_only (B5 end-to-end)        | **0.760**      | **0.802**    |
| combined (deep+texture+saliency) | 0.741          | 0.782        |
| mri\_hybrid (deep+texture)       | 0.733          | 0.774        |
| deep\_only (B5 embeddings)       | 0.722          | 0.761        |
| texture\_only (GLCM+LBP)         | 0.697          | 0.738        |
| saliency\_only                   | 0.648          | 0.685        |

Deep learned features dominate, texture features add complementary
signal on top of the CNN embeddings, and combining all three feature
families plus the end-to-end classifier gives the strongest
subject-level result — majority voting across a subject's slices adds
\~4 points over the per-slice score.

---

## 6. How to run

```powershell
pip install -r requirements.txt
pip install hd-bet            # optional; morphology fallback used if absent

# patient-safe production pipeline (splits, manifests, frozen bundles)
python -m asd_training prepare --config configs/hybrid.json --run-dir runs/hybrid/<run>
python -m asd_training train   --run-dir runs/hybrid/<run>
python -m asd_training report  --run-dir runs/hybrid/<run>

# paper-style slice-level 5-fold CV (the §5 numbers)
python -m asd_training paper --source-run runs/hybrid/<run> `
    --run-dir runs/hybrid/paper --aug-copies 1 --epochs 12 --batch-size 6
```

`paper` writes `metrics.json` (per-fold + pooled + subject-vote metrics),
per-fold predictions CSVs and fine-tuned encoder checkpoints under
`runs/hybrid/paper/folds/`; rerunning the same command resumes completed
folds automatically.

### Live demo — one scan through the full pipeline

```powershell
python demo_asd.py                          # random ASD scan
python demo_asd.py --class control          # random control scan
python demo_asd.py ..\..\treatment\51210_skull_stripped.nii.gz
```

Preprocesses a raw NIfTI (resample → nimble → BET check → 9 axial slices →
saliency), scores every slice with the fine-tuned B5, prints the
per-slice probabilities and saves a poster figure (`demo_result.png`)
with raw slices, saliency maps, CNN inputs and the P(ASD) verdict bar.

---

## 7. Repository layout

```javascript
asd_training/
  preprocessing.py   resample, nimble filter, BET/morphology deskull, slicing
  texture.py         masked GLCM + LBP slice features
  saliency.py        LMS opponent-pyramid saliency + enhancement
  deep.py            frozen EfficientNet encoders
  finetune.py        end-to-end fine-tuned encoder (classifier + embeddings)
  features.py        static feature extraction + per-patient cache
  splits.py          immutable patient-level split registry
  training.py        nested-CV Random Forest training
  paper_eval.py      slice-level CV evaluator (paper protocol)
  cli.py             prepare / train / evaluate-test / report / paper
configs/hybrid.json  full reproducible configuration
docs/                protocol notes
tests/               161-test suite
```

Environment used: Python 3.10 · PyTorch 2.5.1+cu121 · CUDA 12.1 ·
RTX 3050 6 GB · scikit-learn · nibabel · OpenCV · HD-BET 2.0.1.
