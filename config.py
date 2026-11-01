"""Central configuration for the autism-detection pipeline.

Paths can be overridden with environment variables:
    ASD_DATA_ROOT          folder containing control/ and treatment/
    ASD_CONTROL_DIR        folder of control  NIfTI files
    ASD_TREATMENT_DIR      folder of ASD      NIfTI files
    ASD_PREPROCESSED_DIR   output of step2_preprocess.py
"""
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------- paths
DATA_ROOT = os.environ.get(
    "ASD_DATA_ROOT", os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
)
CONTROL_DIR = os.environ.get("ASD_CONTROL_DIR", os.path.join(DATA_ROOT, "control"))
TREATMENT_DIR = os.environ.get("ASD_TREATMENT_DIR", os.path.join(DATA_ROOT, "treatment"))

PREPROCESSED_DIR = os.environ.get(
    "ASD_PREPROCESSED_DIR", os.path.join(THIS_DIR, "preprocessed_dataset")
)
SAVED_MODELS_DIR = os.path.join(THIS_DIR, "saved_models")
RESULTS_DIR = os.path.join(THIS_DIR, "results")
MANIFEST_PATH = os.path.join(THIS_DIR, "manifest.csv")
STEP1_SUMMARY = os.path.join(THIS_DIR, "step1_summary.json")
STEP2_SUMMARY = os.path.join(THIS_DIR, "step2_summary.json")

LABELS = {"control": 0, "treatment": 1}
CLASS_DIRS = {"control": CONTROL_DIR, "treatment": TREATMENT_DIR}

# ---------------------------------------------------------- preprocessing
WINDOW_SIZE = 5            # block size for local noise / edge-density maps
G_FILTER = 0.4             # g in K = g*f + (1-g)*M + T*(f-M)   (paper eq. 1)
CANNY_LOW = 100
CANNY_HIGH = 200
# skull-strip heuristics: a stripped brain has few nonzero voxels and an
# essentially empty outer rim
STRIP_MAX_NONZERO = 0.45   # volume nonzero fraction above -> suspect un-stripped
STRIP_MAX_RIM = 0.15       # border-slab nonzero fraction above -> un-stripped
                           # (stripped ABIDE scans still keep ~1-7% rim voxels;
                           #  an un-stripped T1 rim is far denser)
STRIP_RIM_SLAB = 4         # voxels

# --------------------------------------------------------------- training
BACKBONE = "resnet101"     # or "resnet50"
IMG_SIZE = 224
N_SLICES = 3               # mid-1, mid, mid+1 axial slices per subject
EPOCHS = 30
BATCH = 16
LR = 1e-4
WEIGHT_DECAY = 1e-5
N_FOLDS = 5
SEED = 42
EARLY_STOP_PATIENCE = 10
LR_PATIENCE = 5
LR_FACTOR = 0.2
LR_MIN = 1e-6
THRESHOLD = 0.5
LABEL_SMOOTHING = 0.1
BACKBONE_LR_MULT = 0.1       # layer4 learns at LR * this (head gets full LR)
SLICE_CACHE = os.path.join(THIS_DIR, "slice_cache")
VIEW_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}
SLICE_SELECT = "spread"      # or "area" (max-coverage slices in middle half)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
