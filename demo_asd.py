"""Presentation demo — one MRI through the full ASD pipeline.

    python demo_asd.py                          # random scan
    python demo_asd.py --class treatment        # random ASD scan
    python demo_asd.py path/to/scan.nii.gz
    python demo_asd.py scan.nii.gz --label ASD --out figure.png --no-show

Runs the real pipeline end to end (resample -> nimble -> BET check -> axial
slices -> saliency -> EfficientNet-B5) and renders a poster figure with the
raw slices, saliency maps, enhanced CNN inputs and the final P(ASD) verdict.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from asd_training.config import load_config                     # noqa: E402
from asd_training.finetune import FineTunedEncoder              # noqa: E402
from asd_training.preprocessing import prepare_subject          # noqa: E402
from asd_training.saliency import enhance_slice, saliency_stats  # noqa: E402
from asd_training.texture import slice_texture                  # noqa: E402

MODEL_CANDIDATES = (
    "models/best_encoder.pt",
    "runs/hybrid/paper2/folds/fold5/encoder.pt",
    "runs/hybrid/paper2/folds/fold4/encoder.pt",
    "runs/hybrid/paper2/folds/fold3/encoder.pt",
    "runs/hybrid/paper2/folds/fold2/encoder.pt",
    "runs/hybrid/paper2/folds/fold1/encoder.pt",
    "runs/hybrid/paper/folds/fold1/encoder.pt",
    "runs/hybrid/r4/models/finetuned_encoder.pt",
)


def find_model(explicit):
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            sys.exit(f"model not found: {path}")
        return path
    for rel in MODEL_CANDIDATES:
        path = ROOT / rel
        if path.is_file():
            return path
    sys.exit("no trained encoder found; run `python -m asd_training paper` first "
             "or pass --model <checkpoint.pt>")


def pick_scan(args, settings):
    if args.scan:
        path = Path(args.scan)
        if not path.is_file():
            sys.exit(f"scan not found: {path}")
        label = args.label
        if label is None:
            parent = path.parent.name.lower()
            label = {"control": "control", "treatment": "ASD"}.get(parent)
        return path, label
    folder = {"control": settings.control_dir,
              "treatment": settings.treatment_dir,
              "asd": settings.treatment_dir}[str(args.cls).lower()]
    scans = sorted(folder.glob("*.nii.gz")) or sorted(folder.glob("*.nii"))
    if not scans:
        sys.exit(f"no NIfTI scans found in {folder}")
    rng = np.random.default_rng(args.seed)
    path = scans[int(rng.integers(len(scans)))]
    label = "ASD" if folder == settings.treatment_dir else "control"
    return path, label


def bar(probability, width=26):
    filled = int(round(probability * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def render(path, label, prepared, raws, sal_maps, enhanced, probs, out, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(raws)
    prob = float(np.mean(probs))
    verdict = "ASD" if prob >= 0.5 else "CONTROL"
    color = "#c0392b" if verdict == "ASD" else "#1e8449"
    correct = (label is None or
               (verdict == "ASD") == (label.lower() in ("asd", "treatment", "1")))

    fig = plt.figure(figsize=(n * 1.9 + 2, 9.5), facecolor="#111418")
    grid = GridSpec(4, n, figure=fig, height_ratios=[1, 1, 1, 0.75],
                    hspace=0.18, wspace=0.08,
                    left=0.03, right=0.97, top=0.86, bottom=0.06)

    fig.suptitle("ASD Detection from Structural MRI\n"
                 "nimble filter → BET → axial slices → saliency → EfficientNet-B5",
                 color="white", fontsize=15, fontweight="bold")

    row_titles = ("preprocessed slice", "saliency map", "CNN input (enhanced)")
    for row, images in enumerate((raws, sal_maps, enhanced)):
        for col in range(n):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(images[col], cmap="gray" if row != 1 else "hot")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(row_titles[row], color="#aab2bd",
                              fontsize=9, rotation=90, labelpad=28)
                ax.yaxis.set_label_coords(-0.4, 0.5)
                ax.text(-0.12, 0.5, row_titles[row], transform=ax.transAxes,
                        rotation=90, va="center", color="#aab2bd", fontsize=9)
            if row == 0:
                ax.set_title(f"slice {col + 1}", color="#8b949e", fontsize=9)
            if row == 2:
                p = probs[col]
                ax.set_xlabel(f"P(ASD) = {p:.2f}", fontsize=9,
                              color="#c0392b" if p >= 0.5 else "#1e8449",
                              fontweight="bold")
                ax.xaxis.set_label_coords(0.5, -0.12)

    ax = fig.add_subplot(grid[3, :])
    ax.axis("off")
    truth = "" if label is None else (
        f"    true label: {label.upper()}    "
        f"{'CORRECT' if correct else 'INCORRECT'}")
    ax.text(0.5, 0.78, f"PREDICTION:  {verdict}", transform=ax.transAxes,
            ha="center", fontsize=20, fontweight="bold", color=color)
    ax.barh(0.32, prob, height=0.3, color=color, transform=ax.transAxes)
    ax.barh(0.32, 1.0 - prob, height=0.3, left=prob, color="#30363d",
            transform=ax.transAxes)
    ax.axvline(0.5, color="white", lw=1, ls="--", ymin=0.17, ymax=0.47)
    ax.text(0.5, 0.02, f"P(ASD) = {prob:.3f}{truth}", transform=ax.transAxes,
            ha="center", fontsize=11, color="#c9d1d9")
    ax.set_xlim(0, 1)

    out = Path(out)
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    if show:
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
    plt.close(fig)
    return verdict, prob


def main():
    parser = argparse.ArgumentParser(description="ASD pipeline demo")
    parser.add_argument("scan", nargs="?", help="NIfTI path (default: random scan)")
    parser.add_argument("--class", dest="cls", default="treatment",
                        choices=["control", "treatment", "asd"],
                        help="class folder to sample when no scan is given")
    parser.add_argument("--label", help="true label (ASD/control) — inferred from folder")
    parser.add_argument("--model", help="encoder checkpoint (.pt)")
    parser.add_argument("--config", default=str(ROOT / "configs" / "hybrid.json"))
    parser.add_argument("--out", default=str(ROOT / "demo_result.png"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-show", action="store_true", help="save figure, don't display")
    args = parser.parse_args()

    settings = load_config(args.config)
    model_path = find_model(args.model)
    path, label = pick_scan(args, settings)

    print("=" * 62)
    print(" ASD Screening — structural MRI pipeline")
    print(" EfficientNet-B5 (end-to-end) + GLCM/LBP + saliency + RF")
    print("=" * 62)
    print(f"subject    : {path.name}"
          + (f"   (true label: {label.upper()})" if label else ""))

    t0 = time.perf_counter()
    prepared = prepare_subject(str(path), settings)
    qc = prepared["qc"]
    print(f"[1/4] preprocess   resample 1mm -> nimble filter -> "
          f"deskull({qc['deskull']})   {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    view = settings.views[0]
    pairs = prepared["views"][view]
    raws, sal_maps, enhanced, tex = [], [], [], []
    for gray, mask in pairs:
        enh, sal = enhance_slice(gray, settings.saliency_alpha)
        raws.append(gray); sal_maps.append(sal); enhanced.append(enh)
        tex.append(slice_texture(gray, mask, settings)[0])
    tex = np.stack(tex)
    print(f"[2/4] slices       {len(raws)} {view} slices -> "
          f"36-D GLCM/LBP + 6-D saliency per slice   "
          f"{time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    encoder = FineTunedEncoder.load(str(model_path), settings)
    probs = encoder.slice_probabilities(enhanced)
    encoder.close()
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"[3/4] model        {model_path.name} @ {encoder.image_size}px "
          f"on {device}   {time.perf_counter() - t0:.1f}s")
    print("      p(slice|ASD): " + " ".join(f"{p:.2f}" for p in probs))

    verdict, prob = render(path, label, prepared, raws, sal_maps,
                           enhanced, probs, args.out, not args.no_show)
    print(f"[4/4] decision     {bar(prob)}  P(ASD)={prob:.3f} -> {verdict}")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
