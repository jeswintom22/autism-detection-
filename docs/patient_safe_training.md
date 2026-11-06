# Patient-safe EfficientNet + texture + Random Forest training

This pipeline rebuilds autism-vs-control training with patient-disjoint evaluation. It does not depend on the
legacy step1-step4 scripts, legacy preprocessed volumes, slice caches or checkpoints. Control = 0, ASD = 1; one
prediction per patient.

## Commands

```powershell
python -m asd_training prepare --config configs/hybrid.json --run-dir runs/hybrid/<run-id>
python -m asd_training smoke --source-run runs/hybrid/<run-id> --run-dir "<temporary-smoke-directory>"
python -m asd_training train --run-dir runs/hybrid/<run-id>
python -m asd_training evaluate-test --run-dir runs/hybrid/<run-id> --confirm-final
python -m asd_training report --run-dir runs/hybrid/<run-id>
python run_all.py --config configs/hybrid.json --run-dir runs/hybrid/<run-id>
```

- `prepare` audits the cohort (identity, labels, duplicate file/voxel content, volume quality), writes
  `audit.json`, stops on any blocking error, then creates (or verifies) an immutable split registry at
  `runs/hybrid/protocols/<cohort_hash>/splits.json` and writes development/test manifests. Re-running with a
  different configuration or a drifted cohort is refused; a matching prepared run is reused.
- Duplicate-content policy: rows sharing identical file or voxel hashes are grouped under one `person_id`
  (smallest subject ID). All splits and folds are person-level, so identical scans can never span partitions.
  Same-label groups are recorded as warnings (this cohort: 50279+50286, both UM_1 ASD); groups with conflicting
  label or site remain blocking errors.
- `train` reads only the configuration, splits and the development manifest. It never reads the test manifest.
- `smoke` takes 8 control + 8 ASD NYU development patients from a prepared run, uses only B0, two outer/two inner
  folds, one RF candidate with ten trees, and writes to an independent temporary directory marked
  `purpose=smoke`. It never touches the test manifest or borrows other patients and fails if NYU lacks patients.
- `evaluate-test` requires `--confirm-final`, a frozen bundle manifest and unchanged run hashes. It scores all
  frozen bundles once; repeating it returns the saved result.
- `report` renders `report.md` and `figures/` from stored `report_data.json` only (no refitting, thresholds or
  bootstrap).
- `run` / `run_all.py` perform `prepare` + `train` only. Legacy `run_all.py` options and slice-level splitting are
  rejected.

## Method summary

- Patient split: 20% test / 80% development, stratified by site and label, seed 42, then five outer and three inner
  patient-grouped folds. Feature transforms (PCA, scaling, clinical imputation), RF selection and thresholds are
  fitted only inside the relevant training scope.
- Preprocessing: RAS orientation, 1 mm resampling, per-patient in-mask percentile normalisation, nine slices per
  axial/coronal/sagittal view. No registration, harmonisation or augmentation.
- Features: frozen ImageNet EfficientNet B0/B5 embeddings (mean/std pooled per view, PCA fitted per training
  scope), masked GLCM (42/slice) and uniform LBP (54/slice) pooled to 576 texture features, and age, sex, FIQ with
  missingness flags.
- Model families (same splits): clinical-only, texture-only, deep-only, MRI-only hybrid, and the primary combined
  RF (EfficientNet + texture + clinical). Ablations are fixed-setting comparisons, not independently optimised.

## Run artifacts

`config.json`, `environment.json`, `audit.json`, `source_manifest.csv`, `development_manifest.{json,csv}`,
`test_manifest.{json,csv}`, `splits.json`, `run_state.json` (integrity hashes), `cv/` (metrics, outer
predictions, ablation comparison), `frozen_bundle_manifest.json`, `test/` (metrics, intervals, predictions),
`report_data.json`, `report.md`, `figures/`. Outputs may only be written under `runs/hybrid/` or a temporary
directory, never into raw data or legacy `saved_models*`, `results*`, `slice_cache`, `preprocessed_dataset`.
All run data are git-ignored (`runs/`).

## Limitations

- The primary model uses age, sex and FIQ as well as MRI; do not call it MRI-only. FIQ may be unavailable in a
  scan-only deployment.
- The test set is an internal holdout from a cohort already used in earlier experiments, not an external cohort.
  There is no unseen-site guarantee and no clinical diagnostic validation.
- Exact file/voxel hashing does not detect near-duplicates, relatives or metadata errors.
- RAS orientation plus 1 mm resampling is not anatomical registration.
- The >80% test-accuracy target is an objective, not a guarantee; the actual result is reported either way.
- `REPORT.md` and legacy `results*` are historical and are not comparable with this protocol.
