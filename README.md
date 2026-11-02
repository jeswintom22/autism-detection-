# autism-detection-

End-to-end pipeline for detecting Autism Spectrum Disorder from skull-stripped
structural MRI (sMRI), replicating a staged disaster-project style workflow:

```
control/ + treatment/                     (raw *_skull_stripped.nii.gz, outside repo)
    -> step1_collect_data.py              manifest.csv + step1_summary.json
    -> step2_preprocess.py                preprocessed_dataset/<class>/*_filtered.nii.gz
    -> step3_classify.py                  saved_models/ + results/
    -> demo.py                            P(ASD) for one scan
```

`run_all.py` orchestrates the steps and gates training on a validation check.

**Continuing this work?** Read [`context.md`](context.md) first: history, results, conventions, open items.

## Stages

- **step1_collect_data.py** — scans `control/` + `treatment/` (default
  `..\..\`, override with `ASD_DATA_ROOT`), validates every NIfTI loads,
  writes `manifest.csv` (subject id, label, shape, nonzero fraction) and
  `step1_summary.json`.
- **step2_preprocess.py** — per subject, in parallel (`joblib`):
  skull-strip check (rim + nonzero-fraction heuristics; raw scans are
  stripped via HD-BET if installed, else a morphology fallback), then a
  vectorised adaptive **nimble filter** per axial slice —
  `K = g·f + (1−g)·M + T·(f−M)` with M the 4-neighbour mean and T the
  local Canny edge density damped by local noise (paper eqs. 1–2).
  Output volumes keep affine/header; resumable.
- **step3_classify.py** — PyTorch ResNet101 (`--backbone resnet50` to
  switch), everything frozen except `layer4` + new head, all backbone
  BatchNorm frozen, discriminative LR (layer4 = `--lr` × `--bb-lr-mult`,
  default 0.1), label smoothing (`--smooth 0.1`), train augmentation
  (flip/affine/intensity), eval flip-TTA (`--no-tta` to disable),
  per-fold **decision threshold tuned on the training subjects only**
  (stored in each checkpoint + metrics), `BCEWithLogitsLoss(pos_weight)` +
  ReduceLROnPlateau + early stopping, AMP + channels_last + cudnn
  benchmark on CUDA, slice `.npy` cache (`--cache-dir`).
  **Subject-level** `StratifiedGroupKFold` (5 folds): all slices of a
  subject stay in one fold — no leakage; metrics per subject
  (accuracy / precision / recall / F1 / AUC + confusion matrices),
  matching the paper's Table 2 format.
  `--planes 3` feeds axial/coronal/sagittal slices as image channels
  (2.5-D multi-plane input).
- **demo.py** — loads `saved_models/best_model.pt`, deskulls + filters a
  raw scan in-memory (or reads a `_filtered` one), prints `P(ASD)`.

## Usage

```bash
pip install -r requirements.txt

python run_all.py                                   # full pipeline
python run_all.py --max 5 --folds 2 --epochs 1 --limit 10   # smoke test
python step1_collect_data.py                        # individual steps
python step2_preprocess.py --jobs -1
python step3_classify.py --folds 5 --epochs 30
python step3_classify.py --n-slices 9               # spread over middle third
python step3_classify.py --n-slices 9 --slice-select area  # top-coverage slices
python step3_classify.py --n-slices 9 --view coronal       # single-view: coronal
python step3_classify.py --n-slices 9 --planes 3           # + coronal/sagittal channels
python step3_classify.py --no-augment               # ablation: no augmentation
python step3_classify.py --no-tta                   # ablation: no flip TTA
python step3_classify.py --raw                      # ablation: unfiltered data
python demo.py ..\..\control\50030_skull_stripped.nii.gz
python demo.py --ensemble                           # average all 5 fold models

# per-view ensemble: train axial/coronal/sagittal separately, average probs
python step3_classify.py --n-slices 9 --view axial    --models saved_models_ax --results results_ax
python step3_classify.py --n-slices 9 --view coronal  --models saved_models_co --results results_co
python step3_classify.py --n-slices 9 --view sagittal --models saved_models_sa --results results_sa
python ensemble_eval.py results_ax results_co results_sa --out results_ensemble.json
python step4_fuse.py results_ax results_co results_sa      # + age/sex/FIQ/site
python demo.py --model "saved_models_ax,saved_models_co,saved_models_sa" --ensemble
```

## Layout

```
config.py            paths + hyper-parameters (env-var overridable)
step1_collect_data.py
step2_preprocess.py
step3_classify.py
ensemble_eval.py     average val_probs.json across runs (view ensemble)
step4_fuse.py        per-fold stacker: CNN prob + ABIDE phenotype (auto-downloaded)
run_all.py
demo.py
preprocessed_dataset/   (generated, git-ignored)
saved_models/           (generated, git-ignored)
results/                all_metrics.json, curves, confusion matrices
```

Notes: training is a PyTorch port of the original TensorFlow/Colab script
(subject-level CV, class-balanced loss, mid-3 axial slices at 224x224).
The dataset at `D:\study\autisim\{control,treatment}` is already
skull-stripped — the deskull step auto-detects and only strips raw scans.
