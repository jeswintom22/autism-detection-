"""Step 3 - train ResNet101 (transfer learning) with subject-level 5-fold CV.

PyTorch port of the original TensorFlow/Colab pipeline:
  * 3 axial slices (mid-1, mid, mid+1) per subject, min-max -> 224x224x3 uint8
  * torchvision ResNet101 ImageNet weights; everything frozen except layer4
    + a new head; all BatchNorm2d stay frozen (stable on small medical sets)
  * StratifiedGroupKFold: all slices of a subject stay in one fold -> no
    train/validation leakage; metrics computed per subject by averaging
    slice probabilities
  * AdamW + BCEWithLogits(pos_weight=balanced) + ReduceLROnPlateau +
    early stopping; AMP on CUDA

Reads scans from --data-dir (default preprocessed_dataset/ from step2,
or --raw for the original control/treatment folders - ablation mode).
Writes saved_models/*.pt and results/ (metrics json, curves, matrices).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

_MEAN = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)


# ------------------------------------------------------------------- data
def prep_gray(sl, img_size=config.IMG_SIZE):
    """2-D float slice -> (H, W) uint8, per-slice min-max + resize."""
    s = np.nan_to_num(sl.astype(np.float32))
    rng = s.max() - s.min()
    s = np.zeros_like(s) if rng == 0 else (s - s.min()) / rng
    s = cv2.resize(s, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return (s * 255).astype(np.uint8)


def prep_slice(sl, img_size=config.IMG_SIZE):
    """2-D float slice -> (H, W, 3) uint8, per-slice min-max + resize."""
    return np.repeat(prep_gray(sl, img_size)[..., None], 3, axis=-1)


def slice_indices(depth, n_slices):
    """Slice positions: mid-1/mid/mid+1 for n<=3, else spread over the
    middle third of the volume."""
    mid = depth // 2
    if n_slices <= 3:
        return [min(max(mid + o, 0), depth - 1)
                for o in range(-(n_slices // 2), n_slices // 2 + 1)][:n_slices]
    half = max(depth // 6, n_slices // 2)
    half = min(half, mid, depth - 1 - mid)
    idxs = np.linspace(mid - half, mid + half, n_slices).round().astype(int)
    return [int(min(max(i, 0), depth - 1)) for i in idxs]


def area_slice_indices(data, axis, n_slices):
    """Pick the n_slices positions (ascending) with the most brain coverage
    in the middle half of `axis` - area-based anatomical slice selection."""
    depth = data.shape[axis]
    lo, hi = depth // 4, depth - depth // 4
    if n_slices > hi - lo:
        return slice_indices(depth, n_slices)
    others = tuple(a for a in range(3) if a != axis)
    cov = (data > 0).mean(axis=others)[lo:hi]
    top = np.argsort(cov)[-n_slices:]
    return sorted(int(i + lo) for i in top)


def select_indices(data, axis, n_slices, method):
    if method == "area":
        return area_slice_indices(data, axis, n_slices)
    return slice_indices(data.shape[axis], n_slices)


def load_subject_slices(path, n_slices=config.N_SLICES, img_size=config.IMG_SIZE,
                        planes=1, axis=2, select="spread"):
    """(n_slices, H, W, 3) uint8 for one subject volume.

    planes=1: N slices along `axis` (axial=2/coronal=1/sagittal=0), grey
    channel triplicated.
    planes=3: each image's channels = axial/coronal/sagittal slices."""
    data = nib.load(path).get_fdata(dtype=np.float32)
    if planes == 1:
        stack = np.moveaxis(
            np.take(data, select_indices(data, axis, n_slices, select),
                    axis=axis), axis, 0)
        return np.stack([prep_slice(stack[i], img_size)
                         for i in range(n_slices)])
    # one batched strided read per axis -> (n_slices, d1, d2) each
    axis_stacks = [
        np.moveaxis(np.take(data, select_indices(data, a, n_slices, select),
                            axis=a), a, 0)
        for a in range(3)
    ]
    out = np.empty((n_slices, img_size, img_size, 3), np.uint8)
    for i in range(n_slices):
        for a in range(3):
            out[i, :, :, a] = prep_gray(axis_stacks[a][i], img_size)
    return out


def scan_inputs(root):
    pairs = []
    for class_name, label in config.LABELS.items():
        folder = os.path.join(root, class_name)
        files = [
            os.path.join(r, f)
            for r, _, fs in os.walk(folder)
            for f in fs if f.endswith((".nii", ".nii.gz"))
        ]
        print(f"{class_name}: {len(files)} files in {folder}")
        pairs += [(f, label) for f in sorted(files)]
    return pairs


def _load_one(f, n_slices, img_size, planes, axis, select, cache_dir):
    """Load (or cache+load) the slice stack for one subject; None on failure."""
    try:
        cpath = None
        if cache_dir:
            st = os.stat(f)
            key = hashlib.md5(
                f"{os.path.abspath(f)}|{st.st_mtime}|{st.st_size}".encode()
            ).hexdigest()[:16]
            tag = f"_n{n_slices}_p{planes}"
            if axis != 2 or select != "spread":
                tag += f"_a{axis}_s{select}"
            cpath = os.path.join(cache_dir, f"{key}{tag}.npy")
            if os.path.exists(cpath):
                return np.load(cpath)
        sl = load_subject_slices(f, n_slices, img_size, planes, axis, select)
        if cpath:
            tmp = f"{cpath}.{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                np.save(fh, sl)
            os.replace(tmp, cpath)
        return sl
    except Exception as e:  # noqa: BLE001
        print(f"Failed {f}: {e}")
        return None


def load_dataset(pairs, n_slices, img_size, planes=1, axis=2, select="spread",
                 cache_dir=None, jobs=0):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    if jobs:
        from joblib import Parallel, delayed
        arrays = Parallel(n_jobs=jobs, verbose=5)(
            delayed(_load_one)(f, n_slices, img_size, planes, axis, select,
                               cache_dir)
            for f, _ in pairs)
    else:
        arrays = []
        for i, (f, _) in enumerate(pairs, 1):
            arrays.append(_load_one(f, n_slices, img_size, planes, axis,
                                    select, cache_dir))
            if i % 100 == 0:
                print(f"  loaded {i}/{len(pairs)}")
    X, y, groups, subjects = [], [], [], []
    sid = 0
    for (f, label), sl in zip(pairs, arrays):
        if sl is None:
            continue
        subjects.append(f)
        X.append(sl)
        y.extend([label] * len(sl))
        groups.extend([sid] * len(sl))
        sid += 1
    if not X:
        sys.exit("no slices loaded - check input paths")
    X = np.concatenate(X)
    print(f"X={X.shape}  subjects={sid}  slices={len(y)}")
    return X, np.array(y), np.array(groups), subjects


def augment_slice(img, rng):
    """(H,W,3) uint8 -> augmented copy. L-R flip is valid for axial brain
    slices; affine + intensity jitter add diversity."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
    img = np.ascontiguousarray(img)
    if rng.random() < 0.7:
        h, w = img.shape[:2]
        ang = rng.uniform(-8, 8)
        sc = rng.uniform(0.95, 1.05)
        dx, dy = rng.uniform(-0.05, 0.05, 2) * [w, h]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        m[:, 2] += [dx, dy]
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        a, b = rng.uniform(0.9, 1.1), rng.uniform(-12, 12)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255)
    return np.ascontiguousarray(img.astype(np.uint8))


class SliceDataset(Dataset):
    def __init__(self, X, y, augment=False, flip=False, seed=config.SEED):
        # torch tensor -> dataloader workers share it via shared memory on
        # Windows spawn instead of pickling a full copy per worker
        self.X = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
        self.y, self.augment, self.flip = y, augment, flip
        self.seed, self.epoch = seed, 0

    def set_epoch(self, ep):
        """Per-(epoch, index) deterministic aug seed -> worker/order safe."""
        self.epoch = ep

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.X[i]
        if self.augment:
            img = augment_slice(
                img.numpy(), np.random.default_rng(self.seed + self.epoch * 10007 + i))
        elif self.flip:
            img = img.flip(1)
        x = img.permute(2, 0, 1).float().div_(255.0) \
            if isinstance(img, torch.Tensor) \
            else torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        x = (x - _MEAN) / _STD
        return x, torch.tensor(self.y[i], dtype=torch.float32)


# ------------------------------------------------------------------- model
def build_model(backbone=config.BACKBONE):
    if backbone == "resnet101":
        m = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    elif backbone == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"unknown backbone {backbone}")
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.layer4.parameters():          # ~ last 30 layers trainable
        p.requires_grad_(True)
    for mod in m.modules():                  # keep every backbone BN frozen
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Linear(in_f, 512), nn.SiLU(), nn.BatchNorm1d(512), nn.Dropout(0.5),
        nn.Linear(512, 256), nn.SiLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    return m


def _freeze_bn(model):
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()


# ---------------------------------------------------------------- training
def _move(x, device, pin):
    x = x.to(device, non_blocking=pin)
    return x.contiguous(memory_format=torch.channels_last) if pin else x


def train_fold(model, tr_ds, va_loader, args, device):
    n_pos = float(tr_ds.y.sum())
    pos_weight = torch.tensor((len(tr_ds.y) - n_pos) / max(n_pos, 1.0), device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    crit_val = nn.BCEWithLogitsLoss()
    bb, head = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (bb if n.startswith("layer4") else head).append(p)
    opt = torch.optim.AdamW(
        [{"params": bb, "lr": args.lr * args.bb_lr_mult},
         {"params": head, "lr": args.lr}],
        weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE, min_lr=config.LR_MIN)
    pin = device.type == "cuda"
    tr_loader = DataLoader(
        tr_ds, batch_size=args.batch, shuffle=True, pin_memory=pin,
        num_workers=args.workers, persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None)
    scaler = torch.amp.GradScaler("cuda", enabled=pin)
    best_loss, best_state, bad = float("inf"), None, 0
    hist = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    for ep in range(1, args.epochs + 1):
        model.train()
        _freeze_bn(model)
        tr_ds.set_epoch(ep)
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in tr_loader:
            x = _move(x, device, pin)
            y = y.to(device, non_blocking=pin)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=pin):
                logit = model(x).squeeze(1)
                y_s = y * (1.0 - args.smooth) + 0.5 * args.smooth
                loss = crit(logit, y_s)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * len(y)
            correct += ((logit > 0).float() == y).sum().item()
            n += len(y)

        model.eval()
        vloss, vcorrect, vn = 0.0, 0, 0
        with torch.no_grad():
            for x, y in va_loader:
                x = _move(x, device, pin)
                y = y.to(device, non_blocking=pin)
                with torch.autocast(device.type, enabled=pin):
                    logit = model(x).squeeze(1)
                    vloss += crit_val(logit, y).item() * len(y)
                vcorrect += ((logit > 0).float() == y).sum().item()
                vn += len(y)

        tr_loss, tr_acc = loss_sum / n, correct / n
        va_loss, va_acc = vloss / vn, vcorrect / vn
        hist["loss"].append(tr_loss)
        hist["acc"].append(tr_acc)
        hist["val_loss"].append(va_loss)
        hist["val_acc"].append(va_acc)
        sched.step(va_loss)
        lr_now = opt.param_groups[0]["lr"]
        print(f"  ep {ep:02d} loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val {va_loss:.4f} acc {va_acc:.3f} | lr {lr_now:.1e}")

        if va_loss < best_loss - 1e-4:
            best_loss, bad = va_loss, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= config.EARLY_STOP_PATIENCE:
                print(f"  early stop at epoch {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


@torch.no_grad()
def predict_prob(model, dataset, device, batch):
    model.eval()
    pin = device.type == "cuda"
    loader = DataLoader(dataset, batch_size=batch, pin_memory=pin)
    out = []
    for x, _ in loader:
        x = _move(x, device, pin)
        with torch.autocast(device.type, enabled=pin):
            logit = model(x).squeeze(1)
        out.append(torch.sigmoid(logit.float()).cpu())
    return torch.cat(out).numpy()


def eval_probs(model, ds, device, batch, tta):
    """Slice probabilities, optionally averaged with the L-R-flipped input."""
    p = predict_prob(model, ds, device, batch)
    if tta:
        p = (p + predict_prob(model, SliceDataset(ds.X, ds.y, flip=True),
                              device, batch)) / 2.0
    return p


def best_threshold(y, p):
    """Max-F1 operating point, tuned on TRAIN subjects only."""
    ths = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y, (p > t).astype(int), zero_division=0) for t in ths]
    best = np.flatnonzero(np.isclose(f1s, max(f1s), atol=1e-6))
    return float(ths[int(best[len(best) // 2])])


def subject_level(y, prob, groups):
    ids = np.unique(groups)
    ys = np.array([y[groups == g][0] for g in ids])
    ps = np.array([prob[groups == g].mean() for g in ids])
    return ys, ps


# ---------------------------------------------------------------- plotting
def plot_history(hist, fold, out_dir):
    d = os.path.join(out_dir, "training_metrics")
    os.makedirs(d, exist_ok=True)
    for key, title in [("loss", "Loss"), ("acc", "Accuracy")]:
        plt.figure(figsize=(8, 5))
        plt.plot(hist[key], label=f"Training {title}")
        plt.plot(hist[f"val_{key}"], label=f"Validation {title}")
        plt.title(f"Model {title} - Fold {fold}")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(os.path.join(d, f"fold_{fold}_{key}.png"))
        plt.close()


def plot_cm(cm, title, out_dir, fmt="d"):
    d = os.path.join(out_dir, "confusion_matrices")
    os.makedirs(d, exist_ok=True)
    cm = np.asarray(cm, dtype=np.float64)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            v = int(cm[i, j]) if fmt == "d" else cm[i, j]
            plt.text(j, i, format(v, fmt), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["Control", "ASD"])
    plt.yticks([0, 1], ["Control", "ASD"])
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(d, f"{title.replace(' ', '_')}.png"))
    plt.close()


# ------------------------------------------------------------------ run cv
def save_model(model, path, args, fold, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "backbone": args.backbone,
        "img_size": config.IMG_SIZE,
        "n_slices": args.n_slices,
        "planes": args.planes,
        "axis": getattr(args, "axis", 2),
        "select": getattr(args, "slice_select", "spread"),
        "tta": not args.no_tta,
        "threshold": (metrics or {}).get("threshold", config.THRESHOLD),
        "fold": fold,
        "metrics": metrics,
    }, path)


def run_cv(X, y, groups, args, device, subjects):
    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                               random_state=config.SEED)
    pin = device.type == "cuda"
    folds, cms, best_f1 = {}, [], -1.0
    fold_probs = {}

    for k, (tr, va) in enumerate(skf.split(X, y, groups), 1):
        print(f"\n===== Fold {k}/{args.folds} =====")
        assert not set(groups[tr]) & set(groups[va]), "subject leakage!"
        model = build_model(args.backbone).to(device)
        if pin:
            model = model.to(memory_format=torch.channels_last)
        tr_ds = SliceDataset(X[tr], y[tr], augment=not args.no_augment,
                             seed=config.SEED + k)
        tr_eval = SliceDataset(X[tr], y[tr])
        va_ds = SliceDataset(X[va], y[va])
        va_loader = DataLoader(va_ds, batch_size=args.batch, pin_memory=pin)
        hist = train_fold(model, tr_ds, va_loader, args, device)
        plot_history(hist, k, args.results)

        # operating point tuned on TRAIN subjects only (no val peeking)
        prob_tr = eval_probs(model, tr_eval, device, args.batch, not args.no_tta)
        ys_tr, ps_tr = subject_level(y[tr], prob_tr, groups[tr])
        thr = best_threshold(ys_tr, ps_tr)

        prob = eval_probs(model, va_ds, device, args.batch, not args.no_tta)
        ys, ps = subject_level(y[va], prob, groups[va])
        pred = (ps > thr).astype(int)
        try:
            auc = roc_auc_score(ys, ps)
        except ValueError:
            auc = float("nan")
        m = {
            "accuracy": float(accuracy_score(ys, pred)),
            "precision": float(precision_score(ys, pred, zero_division=0)),
            "recall": float(recall_score(ys, pred, zero_division=0)),
            "f1": float(f1_score(ys, pred, zero_division=0)),
            "auc": float(auc),
            "threshold": thr,
        }
        save_model(model, os.path.join(args.models, f"model_fold_{k}.pt"),
                   args, k, m)
        folds[f"fold_{k}"] = m
        cm = confusion_matrix(ys, pred, labels=[0, 1])
        cms.append(cm)
        plot_cm(cm, f"Fold {k}", args.results)
        print({a: round(b, 4) for a, b in m.items()})

        # per-subject held-out probabilities, joined by basename across runs
        # (e.g. axial/coronal/sagittal) for ensemble_eval.py
        ids = np.unique(groups[va])
        fold_probs[f"fold_{k}"] = {
            "threshold": thr,
            "subjects": [
                {"file": os.path.basename(subjects[g]),
                 "y": int(ys[j]), "p": float(ps[j])}
                for j, g in enumerate(ids)
            ],
        }

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            save_model(model, os.path.join(args.models, "best_model.pt"),
                       args, k, m)

        del model
        if pin:
            torch.cuda.empty_cache()

    overall = {
        mt: {"mean": float(np.nanmean([f[mt] for f in folds.values()])),
             "std": float(np.nanstd([f[mt] for f in folds.values()]))}
        for mt in ["accuracy", "precision", "recall", "f1", "auc"]
    }
    plot_cm(np.mean(cms, axis=0), "Average Confusion Matrix",
            args.results, fmt=".1f")
    os.makedirs(args.results, exist_ok=True)
    with open(os.path.join(args.results, "all_metrics.json"), "w") as f:
        json.dump({"fold_metrics": folds, "overall_metrics": overall}, f, indent=4)
    with open(os.path.join(args.results, "val_probs.json"), "w") as f:
        json.dump(fold_probs, f)
    print("\nOverall (subject-level):")
    for mt, s in overall.items():
        print(f"  {mt}: {s['mean']:.3f} +/- {s['std']:.3f}")


# -------------------------------------------------------------- inference
def predict_slices(model, arr_uint8, device, tta=False):
    """arr: (n_slices,H,W,3) uint8 -> mean P(ASD) over slices (+TTA flip)."""
    model.eval()
    pin = device.type == "cuda"

    def _prob(a):
        x = torch.from_numpy(np.ascontiguousarray(a))
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        x = _move((x - _MEAN) / _STD, device, pin)
        with torch.no_grad():
            return float(torch.sigmoid(model(x).squeeze(1)).mean().item())

    p = _prob(arr_uint8)
    if tta:
        p = (p + _prob(arr_uint8[:, :, ::-1, :])) / 2.0
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=config.PREPROCESSED_DIR,
                    help="root containing <class>/*.nii.gz")
    ap.add_argument("--raw", action="store_true",
                    help="train on the raw control/treatment folders (ablation)")
    ap.add_argument("--backbone", default=config.BACKBONE,
                    choices=["resnet101", "resnet50"])
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch", type=int, default=config.BATCH)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--limit", type=int, default=0,
                    help="use at most N subjects per class (smoke test)")
    ap.add_argument("--n-slices", type=int, default=config.N_SLICES,
                    help="slices/subject; <=3 = mid-adjacent, >3 = spread "
                         "over the middle third")
    ap.add_argument("--planes", type=int, default=1, choices=[1, 3],
                    help="1 = single view; 3 = axial/coronal/sagittal channels")
    ap.add_argument("--view", default="axial",
                    choices=list(config.VIEW_AXIS),
                    help="anatomical plane for planes=1 (train once per view, "
                         "then ensemble subject probs)")
    ap.add_argument("--slice-select", default=config.SLICE_SELECT,
                    choices=["spread", "area"],
                    help="spread = middle-third; area = top coverage slices "
                         "in the middle half")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable train-time augmentation (ablation)")
    ap.add_argument("--no-tta", action="store_true",
                    help="disable flip test-time augmentation at eval")
    ap.add_argument("--smooth", type=float, default=config.LABEL_SMOOTHING,
                    help="label smoothing on train targets")
    ap.add_argument("--bb-lr-mult", type=float, default=config.BACKBONE_LR_MULT,
                    help="layer4 LR = --lr * this (head uses --lr)")
    ap.add_argument("--cache-dir", default=config.SLICE_CACHE,
                    help="slice .npy cache ('' disables)")
    ap.add_argument("--jobs", type=int, default=-1,
                    help="parallel slice loading (joblib workers; 0 = sequential)")
    ap.add_argument("--workers", type=int, default=4,
                    help="dataloader workers for the train loader "
                         "(overlaps CPU aug with GPU compute; 0 disables)")
    ap.add_argument("--models", default=config.SAVED_MODELS_DIR)
    ap.add_argument("--results", default=config.RESULTS_DIR)
    args = ap.parse_args(argv)
    if args.folds < 2:
        ap.error("--folds must be >= 2 (StratifiedGroupKFold)")
    args.axis = config.VIEW_AXIS[args.view]

    root = config.DATA_ROOT if args.raw else args.data_dir
    if not os.path.isdir(os.path.join(root, "control")):
        sys.exit(f"no control/ under {root} - run step2 or pass --raw/--data-dir")

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    if device.type == "cuda":               # free fp32 speedup on RTX 30xx
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print("device:", device, "| backbone:", args.backbone,
          "| data:", os.path.abspath(root),
          f"| slices={args.n_slices} planes={args.planes} view={args.view} "
          f"sel={args.slice_select} "
          f"tta={not args.no_tta} aug={not args.no_augment}")
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    pairs = scan_inputs(root)
    if args.limit:
        by_label = {}
        for p in pairs:
            by_label.setdefault(p[1], []).append(p)
        pairs = [p for lab in by_label.values() for p in lab[: args.limit]]
        print(f"limit: {len(pairs)} subjects")
    if not pairs:
        sys.exit("no scans found")

    t0 = time.time()
    X, y, groups, subjects = load_dataset(
        pairs, args.n_slices, config.IMG_SIZE, planes=args.planes,
        axis=args.axis, select=args.slice_select,
        cache_dir=args.cache_dir or None, jobs=args.jobs)
    print(f"loaded in {time.time() - t0:.0f}s")
    run_cv(X, y, groups, args, device, subjects)
    return 0


if __name__ == "__main__":
    sys.exit(main())
