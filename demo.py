"""Demo - predict P(ASD) for a single NIfTI scan with the saved best model.

    python demo.py                          # first scan in preprocessed/treatment
    python demo.py path\to\scan.nii.gz      # any .nii/.nii.gz (raw or filtered)

Raw inputs are deskull-checked and nimble-filtered in-memory (middle slices
only); files already ending in _filtered.nii.gz are used as-is.
"""
import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np
import torch

import config
import step3_classify as s3


def default_scan():
    folder = os.path.join(config.PREPROCESSED_DIR, "treatment")
    if os.path.isdir(folder):
        files = sorted(f for f in os.listdir(folder)
                       if f.endswith((".nii", ".nii.gz")))
        if files:
            return os.path.join(folder, files[0])
    return None


def subject_input(path, n_slices, planes, axis=2, select="spread",
                  img_size=config.IMG_SIZE):
    """(n_slices,H,W,3) uint8 ready for the model."""
    import step2_preprocess as s2

    data = nib.load(path).get_fdata(dtype=np.float32)
    if not s2.is_stripped(data):
        print("input not skull-stripped - stripping in-memory")
        data, _ = s2.skull_strip(path, data)
    filtered = "_filtered" in os.path.basename(path)

    def _gray(a, idx):
        sl = np.take(data, idx, axis=a)
        if not filtered:
            sl = s2.nimble_filter_slice(sl)
        return s3.prep_gray(sl, img_size)

    if planes == 1:
        idxs = s3.select_indices(data, axis, n_slices, select)
        return np.stack([np.repeat(_gray(axis, z)[..., None], 3, -1)
                         for z in idxs])
    per_axis = [s3.select_indices(data, a, n_slices, select) for a in range(3)]
    return np.stack([np.stack([_gray(a, per_axis[a][i]) for a in range(3)], -1)
                     for i in range(n_slices)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("nii", nargs="?", default=None, help="path to .nii/.nii.gz")
    ap.add_argument("--model", default=os.path.join(config.SAVED_MODELS_DIR,
                                                  "best_model.pt"))
    ap.add_argument("--ensemble", action="store_true",
                    help="average checkpoints; --model may be a dir (uses all "
                         "model_fold_*.pt inside), a file, or a comma-separated "
                         "mix of both")
    args = ap.parse_args(argv)

    path = args.nii or default_scan()
    if not path or not os.path.exists(path):
        sys.exit("no scan given and none found in preprocessed_dataset/treatment")
    if not args.ensemble and not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model} - run step3_classify.py first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.ensemble:
        # --model may be a dir, a file, or a comma-separated mix of both
        paths = []
        for token in args.model.split(","):
            token = token.strip()
            if os.path.isdir(token):
                paths += sorted(glob.glob(os.path.join(token, "model_fold_*.pt")))
            elif os.path.isfile(token):
                paths.append(token)
            else:
                paths += sorted(glob.glob(token))
        if not paths:
            sys.exit(f"no checkpoints resolved from {args.model}")
    else:
        paths = [args.model]

    arrs, probs, thrs = {}, [], []
    for mp in paths:
        ckpt = torch.load(mp, map_location=device, weights_only=True)
        model = s3.build_model(ckpt.get("backbone", config.BACKBONE)).to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        cfg = (ckpt.get("n_slices", config.N_SLICES), ckpt.get("planes", 1),
               ckpt.get("axis", 2), ckpt.get("select", "spread"),
               ckpt.get("img_size", config.IMG_SIZE))
        if cfg not in arrs:
            arrs[cfg] = subject_input(path, *cfg)
        thrs.append(ckpt.get("threshold", config.THRESHOLD))
        p = s3.predict_slices(model, arrs[cfg], device,
                              tta=ckpt.get("tta", False))
        probs.append(p)
        if args.ensemble:
            print(f"  {os.path.basename(mp)}: {p:.3f}")

    p, thr = float(np.mean(probs)), float(np.mean(thrs))
    label = "ASD" if p > thr else "Control"
    print(f"{os.path.basename(path)}: P(ASD)={p:.3f} -> {label} (thr={thr:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
