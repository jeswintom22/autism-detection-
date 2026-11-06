import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .artifacts import atomic_json, sha256_file, stable_hash
from .deep import FrozenEncoder
from .preprocessing import cnn_image


class FineTunedEncoder:
    """EfficientNet fine-tuned per patient fold; embeddings feed the Random Forest."""

    def __init__(self, settings, backbone="efficientnet_b5", allow_download=True, weights_path=None):
        self.name = backbone
        self.settings = settings
        self.embedding_dim = 256
        self.image_size = {"efficientnet_b0": 224, "efficientnet_b5": 456}[backbone]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = FrozenEncoder(backbone, settings, allow_download=allow_download,
                             weights_path=weights_path)
        self.weights_path = base.weights_path
        self.weight_hash = base.weight_hash
        cnn = base.model  # features + avgpool, classifier=Identity -> GAP vector
        head_width = base.dimension
        blocks = list(cnn.features.children())
        trainable = int((settings.finetune or {}).get("unfreeze_blocks", 3))
        for parameter in cnn.parameters():
            parameter.requires_grad_(False)
        for block in blocks[max(0, len(blocks) - trainable):]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        cnn.eval()  # frozen BN statistics; only conv weights in the last blocks update
        self.head = nn.Sequential(
            nn.Linear(head_width, 512), nn.SiLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, self.embedding_dim), nn.SiLU(), nn.BatchNorm1d(self.embedding_dim),
            nn.Dropout(0.3),
        )
        self.logit = nn.Linear(self.embedding_dim, 1)
        self.cnn, self.head, self.logit = cnn.to(self.device), self.head.to(self.device), self.logit.to(self.device)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device)[None, :, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device)[None, :, None, None]

    @classmethod
    def load(cls, path, settings):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        encoder = cls(settings, checkpoint["backbone"], allow_download=False,
                      weights_path=checkpoint.get("imagenet_weights_path"))
        if encoder.weight_hash != checkpoint.get("imagenet_weight_sha256"):
            raise ValueError("Fine-tuned encoder ImageNet weights fingerprint mismatch")
        encoder.cnn.load_state_dict(checkpoint["cnn"])
        encoder.head.load_state_dict(checkpoint["head"])
        encoder.logit.load_state_dict(checkpoint["logit"])
        encoder.cnn.eval().requires_grad_(False)
        encoder.head.eval().requires_grad_(False)
        encoder.logit.eval().requires_grad_(False)
        return encoder

    def metadata(self):
        return {
            "kind": "finetuned", "backbone": self.name, "embedding_dim": self.embedding_dim,
            "image_size": self.image_size, "imagenet_weight_sha256": self.weight_hash,
            "precision": "amp-bf16" if self.device.type == "cuda" and self.settings.amp else "float32",
            "pooling": "slice-embedding-mean-std", "transform": "letterbox-imagenet-v1",
            "input": "saliency-enhanced", "fine_tuned": True,
        }

    def _batch_tensor(self, grays):
        images = np.stack([cnn_image(gray, self.image_size) for gray in grays])
        tensor = torch.from_numpy(images).to(self.device)
        return (tensor - self.mean) / self.std

    def _augment(self, tensor):
        batch = tensor.shape[0]
        angles = (torch.rand(batch, device=self.device) - 0.5) * math.radians(16)
        flips = torch.rand(batch, device=self.device) < 0.5
        cos, sin = torch.cos(angles), torch.sin(angles)
        theta = torch.zeros(batch, 2, 3, device=self.device)
        theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
        theta[:, 1, 0], theta[:, 1, 1] = sin, cos
        tensor = F.grid_sample(
            tensor, F.affine_grid(theta, tensor.shape, align_corners=False),
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )
        return torch.where(flips[:, None, None, None], tensor.flip(-1), tensor)

    def _forward(self, tensor, train=False):
        amp_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        with torch.autocast(self.device.type, dtype=amp_dtype,
                            enabled=self.device.type == "cuda" and self.settings.amp):
            gap = self.cnn(tensor).flatten(1)
        embedding = self.head(gap.float())
        logit = self.logit(embedding).squeeze(1)
        return embedding, logit

    def _step_loss(self, entries, train):
        grays = [gray for _, gray, _ in entries]
        labels = torch.tensor([label for _, _, label in entries],
                              dtype=torch.float32, device=self.device)
        tensor = self._batch_tensor(grays)
        if train:
            tensor = self._augment(tensor)
        _, logit = self._forward(tensor)
        pos_weight = getattr(self, "_pos_weight", None)
        return F.binary_cross_entropy_with_logits(logit, labels, pos_weight=pos_weight)

    @staticmethod
    def _finite(value):
        return float(value) if math.isfinite(float(value)) else None

    @staticmethod
    def _stratified_split(sids, labels, fraction, seed):
        rng = np.random.default_rng(seed)
        train_sids, val_sids = [], []
        for value in (0, 1):
            group = sorted(sid for sid in sids if labels[sid] == value)
            rng.shuffle(group)
            n_val = max(1, int(round(len(group) * fraction)))
            val_sids.extend(group[:n_val])
            train_sids.extend(group[n_val:])
        return sorted(train_sids), sorted(val_sids)

    def finetune(self, patients, labels, config, checkpoint_path, deadline,
                 progress=lambda message: print(message, flush=True)):
        """patients: list of (subject_id, {view: (S,H,W) array}); labels: {sid: 0/1}."""
        cfg = dict({
            "epochs": 10, "batch_size": 4, "lr_backbone": 1e-4, "lr_head": 1e-3,
            "weight_decay": 1e-4, "unfreeze_blocks": 3, "patience": 3, "val_fraction": 0.15,
        }, **(config or {}))
        torch.manual_seed(self.settings.seed)
        sids = sorted(labels)
        if not sids or set(sids) != {sid for sid, _ in patients}:
            raise ValueError("Fine-tune labels must match the provided patients exactly")
        patient_map = {sid: views for sid, views in patients}
        train_sids, val_sids = self._stratified_split(
            sids, labels, cfg["val_fraction"], self.settings.seed)
        if not train_sids or not val_sids or set(train_sids) & set(val_sids):
            raise ValueError("Fine-tune split must be patient-disjoint with both classes")
        entries = []
        for sid in sids:
            for view in self.settings.views:
                for gray in patient_map[sid][view]:
                    entries.append((sid, np.asarray(gray, dtype=np.float32), int(labels[sid])))
        train_entries = [e for e in entries if e[0] in set(train_sids)]
        val_entries = [e for e in entries if e[0] in set(val_sids)]
        if not train_entries or not val_entries:
            raise ValueError("Fine-tune produced empty train/validation slice sets")
        gpu_name = (torch.cuda.get_device_name(self.device)
                    if self.device.type == "cuda" else "cpu")
        progress(f"finetune {self.name}: device={self.device.type} ({gpu_name}) "
                 f"train={len(train_entries)} val={len(val_entries)}")
        backbone_params = [p for p in self.cnn.parameters() if p.requires_grad]
        head_params = list(self.head.parameters()) + list(self.logit.parameters())
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": cfg["lr_backbone"]},
            {"params": head_params, "lr": cfg["lr_head"]},
        ], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=cfg["lr_backbone"] * 0.01)
        rng = np.random.default_rng(self.settings.seed + 1)
        use_amp = self.device.type == "cuda" and self.settings.amp
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        train_label_values = [labels[sid] for sid in train_sids]
        n_pos = sum(train_label_values)
        n_neg = len(train_label_values) - n_pos
        if not n_pos or not n_neg:
            raise ValueError("Fine-tune training split requires both classes")
        self._pos_weight = torch.tensor(n_neg / n_pos, device=self.device)
        best_val, stale = math.inf, 0
        best_state = None
        history = []
        for epoch in range(cfg["epochs"]):
            if time.monotonic() >= deadline:
                raise TimeoutError("Soft budget reached during fine-tuning; checkpoint preserved")
            order = rng.permutation(len(train_entries))
            self.head.train(); self.logit.train()  # cnn stays in eval (frozen BN stats)
            total = 0.0
            for start in range(0, len(order), cfg["batch_size"]):
                batch = [train_entries[i] for i in order[start:start + cfg["batch_size"]]]
                optimizer.zero_grad(set_to_none=True)
                loss = self._step_loss(batch, train=True)
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    backbone_params + head_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach()) * len(batch)
            scheduler.step()
            self.cnn.eval(); self.head.eval(); self.logit.eval()
            val_total, val_n = 0.0, 0
            with torch.inference_mode():
                for start in range(0, len(val_entries), cfg["batch_size"] * 2):
                    batch = val_entries[start:start + cfg["batch_size"] * 2]
                    val_total += float(self._step_loss(batch, train=False)) * len(batch)
                    val_n += len(batch)
            val_loss = val_total / max(1, val_n)
            history.append({"epoch": epoch + 1,
                            "train_loss": self._finite(total / len(train_entries)),
                            "val_loss": self._finite(val_loss)})
            progress(f"finetune {self.name}: epoch {epoch + 1}/{cfg['epochs']} "
                     f"train_loss={total / len(train_entries):.4f} val_loss={val_loss:.4f}")
            if val_loss < best_val - 1e-4:
                best_val, stale = val_loss, 0
                best_state = {
                    "cnn": {k: v.detach().cpu().clone() for k, v in self.cnn.state_dict().items()},
                    "head": {k: v.detach().cpu().clone() for k, v in self.head.state_dict().items()},
                    "logit": {k: v.detach().cpu().clone() for k, v in self.logit.state_dict().items()},
                }
            else:
                stale += 1
                if stale >= cfg["patience"]:
                    progress(f"finetune {self.name}: early stop at epoch {epoch + 1}")
                    break
        if best_state is not None:
            self.cnn.load_state_dict(best_state["cnn"])
            self.head.load_state_dict(best_state["head"])
            self.logit.load_state_dict(best_state["logit"])
        self.cnn.eval().requires_grad_(False)
        self.head.eval().requires_grad_(False)
        self.logit.eval().requires_grad_(False)
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backbone": self.name, "imagenet_weights_path": str(self.weights_path),
            "imagenet_weight_sha256": self.weight_hash,
            "cnn": self.cnn.state_dict(), "head": self.head.state_dict(),
            "logit": self.logit.state_dict(),
            "config": cfg, "train_ids": sorted(train_sids), "val_ids": sorted(val_sids),
            "history": history, "settings_finetune": self.settings.finetune,
        }
        payload["data_hash"] = stable_hash({
            "train_ids": payload["train_ids"], "val_ids": payload["val_ids"],
            "labels": {sid: int(labels[sid]) for sid in sids},
            "config": cfg, "seed": self.settings.seed,
        })
        temporary = checkpoint_path.with_name(f"{checkpoint_path.name}.{os.getpid()}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint_path)
        meta = {"file": str(checkpoint_path), "sha256": sha256_file(checkpoint_path),
                "data_hash": payload["data_hash"], "train_ids": payload["train_ids"],
                "val_ids": payload["val_ids"], "history": history,
                "labels_hash": stable_hash({sid: int(labels[sid]) for sid in sids}),
                "metadata": self.metadata()}
        atomic_json(checkpoint_path.with_suffix(".json"), meta)
        return meta

    @torch.inference_mode()
    def _embed_grays(self, grays):
        outputs = []
        batch = max(1, self.settings.batch_sizes.get(self.name, 2))
        for start in range(0, len(grays), batch):
            tensor = self._batch_tensor(grays[start:start + batch])
            embedding, _ = self._forward(tensor)
            outputs.append(embedding.float().cpu().numpy())
        return np.concatenate(outputs)

    def subject_features(self, cnn_views):
        """cnn_views: {view: (n_slices, H, W) saliency-enhanced grayscale slices."""
        pooled = []
        for view in self.settings.views:
            array = np.asarray(cnn_views[view])
            if array.ndim != 3 or array.shape[0] < 1:
                raise ValueError(f"Invalid cached CNN slices for view {view}")
            embeddings = self._embed_grays([np.asarray(gray, dtype=np.float32) for gray in array])
            pooled.extend((embeddings.mean(0), embeddings.std(0)))
        vector = np.concatenate(pooled).astype(np.float32)
        expected = 2 * self.embedding_dim * len(self.settings.views)
        if vector.shape != (expected,) or not np.isfinite(vector).all():
            raise ValueError("Invalid fine-tuned patient embedding")
        return vector

    @torch.inference_mode()
    def slice_probabilities(self, grays):
        """Per-slice ASD probabilities from the logit head (direct classifier use)."""
        outputs = []
        batch = max(1, self.settings.batch_sizes.get(self.name, 2))
        for start in range(0, len(grays), batch):
            tensor = self._batch_tensor(grays[start:start + batch])
            _, logit = self._forward(tensor)
            outputs.append(torch.sigmoid(logit).float().cpu().numpy())
        return np.concatenate(outputs).astype(np.float32)

    def close(self):
        self.cnn.to("cpu"); self.head.to("cpu"); self.logit.to("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
