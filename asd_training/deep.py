from pathlib import Path
import re
from urllib.parse import urlparse

import numpy as np
import torch
from torchvision import models

from .artifacts import sha256_file
from .preprocessing import cnn_image


class FrozenEncoder:
    def __init__(self, backbone, settings, allow_download=True, weights_path=None):
        definitions = {
            "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280, 224),
            "efficientnet_b5": (models.efficientnet_b5, models.EfficientNet_B5_Weights.IMAGENET1K_V1, 2048, 456),
        }
        if backbone not in definitions:
            raise ValueError(f"Unsupported EfficientNet encoder: {backbone}")
        factory, weights, self.dimension, self.image_size = definitions[backbone]
        self.name, self.settings = backbone, settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = settings.batch_sizes[backbone]
        torch.manual_seed(settings.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.weights_path = Path(weights_path) if weights_path else (
            Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
        )
        if not self.weights_path.is_file():
            if weights_path is not None or not allow_download:
                raise FileNotFoundError(f"Verified ImageNet weights unavailable: {self.weights_path}")
            weights.get_state_dict(progress=True, check_hash=True)
        self.weight_hash = sha256_file(self.weights_path)
        if weights_path is None:
            expected_hash = re.search(r"-([a-f0-9]{8,})\.pth$", Path(urlparse(weights.url).path).name)
            if expected_hash is None or not self.weight_hash.startswith(expected_hash.group(1)):
                raise ValueError("EfficientNet checkpoint does not match the official ImageNet weight hash")
        self.model = factory(weights=None)
        try:
            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
        except Exception as exc:
            raise ValueError(f"EfficientNet checkpoint is not a verified {self.name} state dict: {exc}") from exc
        self.model.classifier = torch.nn.Identity()
        self.model.requires_grad_(False).eval().to(self.device)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device)[None, :, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device)[None, :, None, None]

    def metadata(self):
        return {
            "backbone": self.name, "dimension": self.dimension, "image_size": self.image_size,
            "weight_sha256": self.weight_hash, "weights_path": str(self.weights_path),
            "precision": "amp-fp16" if self.device.type == "cuda" and self.settings.amp else "float32",
            "frozen": True, "pooling": "conv-map-spatial-mean-std",
            "transform": "letterbox-imagenet-v1", "input": "saliency-enhanced",
        }

    @torch.inference_mode()
    def _view_features(self, pairs, batch):
        outputs = []
        for start in range(0, len(pairs), batch):
            images = np.stack([cnn_image(gray, self.image_size) for gray, _ in pairs[start:start + batch]])
            tensor = torch.from_numpy(images).to(self.device)
            tensor = (tensor - self.mean) / self.std
            with torch.autocast(self.device.type, enabled=self.device.type == "cuda" and self.settings.amp):
                feature_map = self.model.features(tensor)
            flat = feature_map.flatten(2)
            pooled = torch.cat([flat.mean(2), flat.std(2)], dim=1)
            if pooled.ndim != 2 or pooled.shape[1] != 2 * self.dimension:
                raise ValueError("EfficientNet returned an invalid conv-map embedding shape")
            outputs.append(pooled.float().cpu().numpy())
        return np.concatenate(outputs)

    def subject_features(self, prepared):
        pooled = []
        for view in self.settings.views:
            pairs = prepared.get("cnn_views", prepared["views"])[view]
            try:
                values = self._view_features(pairs, self.batch_size)
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda" or self.batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                self.batch_size = 1
                print(f"{self.name}: CUDA OOM, retrying at batch 1", flush=True)
                values = self._view_features(pairs, 1)
            pooled.extend((values.mean(0), values.std(0)))
        vector = np.concatenate(pooled).astype(np.float32)
        expected = 4 * self.dimension * len(self.settings.views)
        if vector.shape != (expected,) or not np.isfinite(vector).all():
            raise ValueError("Invalid frozen patient embedding")
        return vector

    def close(self):
        self.model.to("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
