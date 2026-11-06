import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ALL_VIEWS = ("axial", "coronal", "sagittal")


@dataclass
class Settings:
    repo_root: Path = ROOT
    control_dir: Path = ROOT.parent.parent / "control"
    treatment_dir: Path = ROOT.parent.parent / "treatment"
    seed: int = 42
    test_size: float = 0.2
    outer_folds: int = 5
    inner_folds: int = 3
    n_slices: int = 9
    spacing: float = 1.0
    margin_mm: int = 8
    percentiles: tuple = (1.0, 99.0)
    views: tuple = ("sagittal",)
    glcm_levels: int = 8
    glcm_distances: tuple = (1,)
    glcm_angles: tuple = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)
    glcm_properties: tuple = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
    lbp_points: tuple = (8, 16)
    lbp_radii: tuple = (1, 2)
    lbp_bins: int = 10
    nimble_window: int = 5
    nimble_g: float = 0.4
    canny_low: int = 100
    canny_high: int = 200
    strip_max_nonzero: float = 0.45
    strip_max_rim: float = 0.15
    strip_rim_slab: int = 4
    saliency_alpha: float = 0.5
    backbones: tuple = ("efficientnet_b0", "efficientnet_b5")
    finetune: dict = field(default_factory=lambda: {"enabled": False})
    extract_workers: int = 0
    pca_components: int = 128
    search_iterations: int = 8
    n_estimators: int = 500
    rf_jobs: int = 4
    batch_sizes: dict = field(default_factory=lambda: {
        "efficientnet_b0": 8, "efficientnet_b5": 2,
    })
    soft_budget_hours: float = 8.0
    amp: bool = True
    purpose: str = "production"
    schema_version: int = 2
    preprocessing_version: str = "ras1mm-nimble-saliency-v3"

    def to_dict(self):
        return {
            f.name: str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v
            for f in dataclasses.fields(self) for v in [getattr(self, f.name)]
        }


def _validate(settings):
    if settings.purpose not in ("production", "smoke"):
        raise ValueError("purpose must be production or smoke")
    if settings.seed != 42 or settings.test_size != 0.2:
        raise ValueError("This protocol fixes seed=42 and patient test_size=0.2")
    if settings.purpose == "production" and (settings.outer_folds, settings.inner_folds) != (5, 3):
        raise ValueError("Production requires five outer and three inner patient folds")
    for name in ("outer_folds", "inner_folds", "n_slices", "pca_components",
                 "search_iterations", "n_estimators", "rf_jobs", "nimble_window",
                 "canny_low", "canny_high", "strip_rim_slab", "lbp_bins"):
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if min(settings.outer_folds, settings.inner_folds) < 2:
        raise ValueError("Patient cross-validation requires at least two folds")
    if not settings.views or set(settings.views) - set(ALL_VIEWS):
        raise ValueError("views must be a non-empty subset of axial/coronal/sagittal")
    if not settings.backbones or set(settings.backbones) - {"efficientnet_b0", "efficientnet_b5"}:
        raise ValueError("Only frozen EfficientNet B0/B5 encoders are supported")
    if len(set(settings.backbones)) != len(settings.backbones):
        raise ValueError("Duplicate backbones")
    if settings.spacing != 1.0 or settings.glcm_levels != 8:
        raise ValueError("This texture protocol requires 1 mm spacing and 8 gray levels")
    if (settings.glcm_distances != (1,)
            or settings.glcm_angles != (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)
            or settings.glcm_properties != ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
            or settings.lbp_points != (8, 16) or settings.lbp_radii != (1, 2)):
        raise ValueError("Texture neighborhoods differ from the approved feature schema")
    if len(settings.percentiles) != 2 or not 0 <= settings.percentiles[0] < settings.percentiles[1] <= 100:
        raise ValueError("Invalid normalization percentiles")
    if not 0 < settings.nimble_g <= 1 or not 0 <= settings.saliency_alpha <= 1:
        raise ValueError("nimble_g and saliency_alpha must lie in (0, 1]")
    if not 0 <= settings.strip_max_nonzero <= 1 or not 0 <= settings.strip_max_rim <= 1:
        raise ValueError("Invalid skull-strip thresholds")
    if settings.margin_mm < 0 or not math.isfinite(settings.soft_budget_hours) or settings.soft_budget_hours <= 0:
        raise ValueError("Invalid crop margin or study budget")
    if any(not isinstance(settings.batch_sizes.get(b), int) or settings.batch_sizes[b] < 1
           for b in settings.backbones):
        raise ValueError("Every encoder requires a positive inference batch size")
    finetune = settings.finetune or {}
    if not isinstance(finetune, dict):
        raise ValueError("finetune must be a mapping")
    if finetune.get("enabled"):
        backbone = finetune.get("backbone", "efficientnet_b5")
        if backbone not in {"efficientnet_b0", "efficientnet_b5"}:
            raise ValueError("finetune.backbone must be efficientnet_b0 or efficientnet_b5")
        for name in ("epochs", "batch_size", "unfreeze_blocks", "patience"):
            value = finetune.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"finetune.{name} must be a positive integer")
        for name in ("lr_backbone", "lr_head", "weight_decay", "val_fraction"):
            value = finetune.get(name)
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)
                                      or not math.isfinite(value) or value <= 0):
                raise ValueError(f"finetune.{name} must be a positive finite number")
        val_fraction = finetune.get("val_fraction", 0.15)
        if not 0 < val_fraction < 0.5:
            raise ValueError("finetune.val_fraction must lie in (0, 0.5)")
    return settings


def settings_from_dict(payload):
    fields = {f.name for f in dataclasses.fields(Settings)}
    unknown = set(payload) - fields
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    values = dict(payload)
    for name in ("repo_root", "control_dir", "treatment_dir"):
        if name in values:
            values[name] = Path(values[name]).expanduser().resolve()
    for name in ("percentiles", "glcm_distances", "glcm_angles", "glcm_properties",
                 "lbp_points", "lbp_radii", "views", "backbones"):
        if name in values:
            values[name] = tuple(values[name])
    return _validate(Settings(**values))


def load_config(path=None):
    source = Path(path) if path is not None else ROOT / "configs" / "hybrid.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    data_root = payload.pop("data_root", "../..")
    root = Path(payload.get("repo_root", ROOT)).expanduser().resolve()

    def resolve(value):
        value = Path(value).expanduser()
        return (value if value.is_absolute() else root / value).resolve()

    data_root = resolve(os.environ.get("ASD_DATA_ROOT", data_root))
    payload["repo_root"] = root
    payload["control_dir"] = resolve(os.environ.get(
        "ASD_CONTROL_DIR", payload.get("control_dir", data_root / "control")))
    payload["treatment_dir"] = resolve(os.environ.get(
        "ASD_TREATMENT_DIR", payload.get("treatment_dir", data_root / "treatment")))
    return settings_from_dict(payload)


def config_hash(settings):
    import hashlib
    return hashlib.sha256(json.dumps(
        settings.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
