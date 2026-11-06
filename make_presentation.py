"""Generate the project presentation deck (runs once)."""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

NAVY = RGBColor(0x12, 0x2B, 0x4A)
NAVY2 = RGBColor(0x1B, 0x3F, 0x6E)
TEAL = RGBColor(0x16, 0xA3, 0xA5)
ORANGE = RGBColor(0xE8, 0x8C, 0x2A)
GREEN = RGBColor(0x2F, 0x9E, 0x5F)
GRAY = RGBColor(0x5A, 0x64, 0x72)
LIGHT = RGBColor(0xF4, 0xF7, 0xFB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x1F, 0x2A, 0x37)

FIG = "runs/hybrid/r2/figures"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height


def slide():
    return prs.slides.add_slide(BLANK)


def rect(s, x, y, w, h, fill, line=None):
    sh = s.shapes.add_shape(1, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    sh.shadow.inherit = False
    return sh


def text(s, x, y, w, h, content, size=18, color=DARK, bold=False, align=PP_ALIGN.LEFT,
         font="Calibri", anchor=MSO_ANCHOR.TOP, line_spacing=1.0):
    box = s.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    if isinstance(content, str):
        content = [content]
    for i, item in enumerate(content):
        if isinstance(item, str):
            item = [(item, {})]
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = line_spacing
        para.space_after = Pt(4)
        for run_text, opts in item:
            run = para.add_run()
            run.text = run_text
            run.font.size = Pt(opts.get("size", size))
            run.font.bold = opts.get("bold", bold)
            run.font.color.rgb = opts.get("color", color)
            run.font.name = font
    return box


def bullets(s, x, y, w, h, items, size=17, gap=6, color=DARK):
    box = s.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        level, txt = item if isinstance(item, tuple) else (0, item)
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.level = level
        para.space_after = Pt(gap)
        para.line_spacing = 1.05
        marker = "\u25aa " if level == 0 else "\u2013 "
        run = para.add_run()
        run.text = marker + txt
        run.font.size = Pt(size - level)
        run.font.color.rgb = color
        run.font.name = "Calibri"
    return box


def title_bar(s, title, subtitle=None, accent=TEAL):
    rect(s, 0, 0, W, Inches(1.02), NAVY)
    rect(s, 0, Inches(1.02), W, Emu(38100), accent)
    text(s, Inches(0.55), Inches(0.12), Inches(12.2), Inches(0.6), title,
         size=28, color=WHITE, bold=True)
    if subtitle:
        text(s, Inches(0.55), Inches(0.62), Inches(12.2), Inches(0.35), subtitle,
             size=13, color=RGBColor(0xBF, 0xD5, 0xEA))


def table(s, x, y, w, rows, col_widths, header_fill=NAVY, size=13, row_h=0.34):
    n_rows, n_cols = len(rows), len(rows[0])
    tbl = s.shapes.add_table(n_rows, n_cols, x, y, w, Inches(row_h * n_rows)).table
    for i, cw in enumerate(col_widths):
        tbl.columns[i].width = Inches(cw)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(val)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.CENTER if r == 0 or c > 0 else PP_ALIGN.LEFT
            run = para.runs[0]
            run.font.size = Pt(size)
            run.font.name = "Calibri"
            run.font.bold = (r == 0)
            run.font.color.rgb = WHITE if r == 0 else DARK
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if r == 0 else (LIGHT if r % 2 else WHITE)
    return tbl


def chip(s, x, y, w, h, value, label, color=TEAL):
    rect(s, x, y, w, h, LIGHT)
    rect(s, x, y, Emu(38100), h, color)
    text(s, x + Inches(0.15), y + Inches(0.08), w - Inches(0.2), Inches(0.5),
         value, size=24, color=NAVY, bold=True)
    text(s, x + Inches(0.15), y + Inches(0.55), w - Inches(0.2), Inches(0.5),
         label, size=11, color=GRAY)


def arrow_between(s, x, y, w=Inches(0.42), h=Inches(0.42)):
    sh = s.shapes.add_shape(13, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = TEAL
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def box_label(s, x, y, w, h, title, sub=None, fill=NAVY2, tcolor=WHITE):
    rect(s, x, y, w, h, fill)
    text(s, x, y + Inches(0.08), w, Inches(0.5), title, size=15, color=tcolor,
         bold=True, align=PP_ALIGN.CENTER)
    if sub:
        text(s, x + Inches(0.08), y + Inches(0.42), w - Inches(0.16), h - Inches(0.45),
             sub, size=10.5, color=RGBColor(0xC8, 0xDA, 0xEE), align=PP_ALIGN.CENTER)


# ---------------- Slide 1: Title ----------------
s = slide()
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, Inches(4.9), W, Emu(38100), TEAL)
text(s, Inches(0.9), Inches(1.9), Inches(11.5), Inches(1.4),
     "Autism Spectrum Disorder Detection\nfrom Structural MRI",
     size=40, color=WHITE, bold=True)
text(s, Inches(0.9), Inches(3.35), Inches(11.5), Inches(1.0),
     "A hybrid machine learning pipeline\n"
     "EfficientNet deep features  +  GLCM / LBP texture  +  clinical data  \u2192  Random Forest",
     size=18, color=RGBColor(0xBF, 0xD5, 0xEA))
text(s, Inches(0.9), Inches(5.3), Inches(11.5), Inches(1.0),
     ["ABIDE structural MRI cohort  \u00b7  patient-level classification",
      "Mini Project \u00b7 2026"],
     size=14, color=GRAY)

# ---------------- Slide 2: Introduction ----------------
s = slide()
title_bar(s, "Introduction", "Why machine learning on structural MRI?")
bullets(s, Inches(0.6), Inches(1.5), Inches(7.0), Inches(5.4), [
    "Autism Spectrum Disorder (ASD) is a neurodevelopmental condition; early, objective detection supports timely intervention.",
    "Diagnosis today relies on behavioral assessment \u2014 subjective, time-consuming, and variable between clinicians.",
    "Structural MRI captures subtle neuroanatomical differences associated with ASD.",
    "Machine learning can learn patterns in MRI that are difficult to observe directly.",
    "Goal: an automated classifier that outputs P(ASD) for each scanned patient.",
])
rect(s, Inches(8.0), Inches(1.5), Inches(4.8), Inches(5.2), LIGHT)
rect(s, Inches(8.0), Inches(1.5), Emu(38100), Inches(5.2), TEAL)
text(s, Inches(8.25), Inches(1.7), Inches(4.3), Inches(0.5), "Project objectives", size=18, color=NAVY, bold=True)
bullets(s, Inches(8.25), Inches(2.3), Inches(4.3), Inches(4.4), [
    "Combine deep and handcrafted image features",
    "Add approved clinical inputs (age, sex, FIQ)",
    "Patient-level classification \u2014 one decision per person",
    "Rigorous, reproducible evaluation protocol",
], size=14, gap=10)

# ---------------- Slide 3: Dataset ----------------
s = slide()
title_bar(s, "Dataset \u2014 ABIDE structural MRI", "Skull-stripped T1-weighted scans")
chip(s, Inches(0.6), Inches(1.5), Inches(2.9), Inches(1.15), "1,004", "MRI scans")
chip(s, Inches(3.7), Inches(1.5), Inches(2.9), Inches(1.15), "504 / 500", "control / ASD")
chip(s, Inches(6.8), Inches(1.5), Inches(2.9), Inches(1.15), "20+", "acquisition sites")
chip(s, Inches(9.9), Inches(1.5), Inches(2.9), Inches(1.15), "803 / 201", "development / test")
table(s, Inches(0.6), Inches(3.1), Inches(7.4), [
    ["Property", "Details"],
    ["Modality", "Skull-stripped T1-weighted structural MRI (.nii.gz)"],
    ["Classes", "Control (0) vs ASD (1) \u2014 nearly balanced"],
    ["Phenotype inputs used", "Age, sex, FIQ (allowlisted; no diagnostic scores)"],
    ["Acquisition", "Multi-site \u2014 mixed orientations and voxel spacings"],
    ["Split", "80 / 20 patient-level, stratified by site and label"],
], [2.4, 5.0], size=12.5)
rect(s, Inches(8.3), Inches(3.1), Inches(4.5), Inches(3.9), LIGHT)
rect(s, Inches(8.3), Inches(3.1), Emu(38100), Inches(3.9), ORANGE)
text(s, Inches(8.55), Inches(3.3), Inches(4.1), Inches(0.5), "Data quality checks", size=18, color=NAVY, bold=True)
bullets(s, Inches(8.55), Inches(3.9), Inches(4.0), Inches(2.8), [
    "All IDs verified against the phenotype table",
    "Valid 3-D geometry confirmed for every scan",
    "File- and voxel-level fingerprints recorded",
    "Missing clinical values flagged explicitly",
], size=13.5, gap=8)

# ---------------- Slide 4: Method overview ----------------
s = slide()
title_bar(s, "Method overview", "From raw MRI to one decision per patient")
y = Inches(2.6)
box_label(s, Inches(0.5), y, Inches(2.1), Inches(1.5), "Skull-stripped\nT1 MRI (.nii.gz)", "mixed spacings")
arrow_between(s, Inches(2.68), y + Inches(0.55))
box_label(s, Inches(3.2), y, Inches(2.2), Inches(1.5), "Preprocessing", "RAS \u00b7 1 mm iso \u00b7 intensity norm \u00b7 9 slices \u00d7 3 views")
arrow_between(s, Inches(5.48), y + Inches(0.55))
for i, (t, sub, col) in enumerate([
    ("EfficientNet B0/B5", "ImageNet embeddings\nper slice \u2192 pooled", NAVY2),
    ("GLCM + LBP texture", "energy, homogeneity,\ncontrast, entropy \u2026", TEAL),
    ("Clinical", "age \u00b7 sex \u00b7 FIQ", ORANGE)]):
    box_label(s, Inches(6.0), Inches(1.45) + i * Inches(1.75), Inches(2.6), Inches(1.55), t, sub, fill=col)
arrow_between(s, Inches(8.68), y + Inches(0.55))
box_label(s, Inches(9.2), y, Inches(2.0), Inches(1.5), "Random Forest\n500 trees", "hyperparameter search + probability threshold")
arrow_between(s, Inches(11.28), y + Inches(0.55))
box_label(s, Inches(11.75), y, Inches(1.4), Inches(1.5), "P(ASD)\nper patient", "", fill=GREEN)
text(s, Inches(0.6), Inches(6.5), Inches(12.2), Inches(0.6),
     "Three complementary views of each patient are fused into a single classifier.",
     size=15, color=GRAY, align=PP_ALIGN.CENTER)

# ---------------- Slide 5: Preprocessing ----------------
s = slide()
title_bar(s, "MRI preprocessing", "Standardizing every scan before feature extraction")
table(s, Inches(0.6), Inches(1.5), Inches(7.4), [
    ["Step", "Specification"],
    ["Orientation", "Canonicalize every scan to RAS"],
    ["Resampling", "Axis-aligned 1 mm isotropic grid (linear image / nearest mask)"],
    ["Brain mask", "Foreground from skull-stripped voxels, re-applied after resampling"],
    ["Normalization", "Per-patient in-mask 1st\u201399th percentile \u2192 clip [0, 1]"],
    ["Slice selection", "9 slices \u00d7 axial / coronal / sagittal, central 20\u201380% extent"],
    ["Quality gates", "\u2265 256 brain pixels/slice; failures reported explicitly"],
    ["CNN input", "Aspect-preserving letterbox \u2192 224 (B0) / 456 (B5)"],
], [1.9, 5.5], size=12.5)
rect(s, Inches(8.3), Inches(1.5), Inches(4.5), Inches(5.3), LIGHT)
rect(s, Inches(8.3), Inches(1.5), Emu(38100), Inches(5.3), TEAL)
text(s, Inches(8.55), Inches(1.7), Inches(4.1), Inches(0.5), "Design choices", size=18, color=NAVY, bold=True)
bullets(s, Inches(8.55), Inches(2.3), Inches(4.0), Inches(4.4), [
    "Fresh processing from original scans \u2014 no stale caches",
    "Slices are sampled across axial, coronal and sagittal anatomy",
    "Intensity scale normalized inside the brain mask only",
    "Deterministic recipe \u2014 every scan gets the identical pipeline",
], size=13.5, gap=8)

# ---------------- Slide 6: Deep features ----------------
s = slide()
title_bar(s, "Deep features \u2014 EfficientNet", "Transfer learning from ImageNet")
bullets(s, Inches(0.6), Inches(1.5), Inches(6.7), Inches(5.4), [
    "Pretrained EfficientNet used as a fixed feature extractor (B0 first, B5 compared).",
    "Classifier head removed \u2192 global convolutional embedding per slice:",
    (1, "1,280 dimensions (B0)  \u00b7  2,048 dimensions (B5)"),
    "27 slices per patient (9 \u00d7 3 views) \u2192 mean + std pooled per view:",
    (1, "7,680 (B0) / 12,288 (B5) deep features per patient"),
    "Fold-local standardization + PCA \u2192 128 compact deep features.",
    "Grayscale replicated to 3 channels, ImageNet mean/std normalization.",
])
rect(s, Inches(7.7), Inches(1.5), Inches(5.1), Inches(5.3), LIGHT)
rect(s, Inches(7.7), Inches(1.5), Emu(38100), Inches(5.3), NAVY2)
text(s, Inches(7.95), Inches(1.7), Inches(4.6), Inches(0.5), "Why EfficientNet?", size=18, color=NAVY, bold=True)
bullets(s, Inches(7.95), Inches(2.3), Inches(4.6), Inches(4.4), [
    "Compound-scaled architecture \u2014 strong accuracy per parameter",
    "ImageNet features transfer well as generic image descriptors",
    "Frozen encoder \u2192 reproducible, fast, no GPU-heavy training",
    "Two sizes compared fairly on the same folds",
], size=13.5, gap=10)

# ---------------- Slide 7: Texture + clinical ----------------
s = slide()
title_bar(s, "Handcrafted + clinical features", "Interpretable descriptors alongside deep features")
box_label(s, Inches(0.6), Inches(1.5), Inches(5.9), Inches(0.9), "GLCM texture (Gray-Level Co-occurrence)", "", TEAL)
bullets(s, Inches(0.6), Inches(2.55), Inches(5.9), Inches(2.2), [
    "Contrast \u00b7 dissimilarity \u00b7 homogeneity \u00b7 ASM \u00b7 energy \u00b7 correlation \u00b7 entropy",
    "32 intensity levels, distances 1/2/4, four angles, brain-mask pairs only",
    "42 features per slice",
], size=13.5)
box_label(s, Inches(0.6), Inches(4.95), Inches(5.9), Inches(0.9), "LBP texture (Local Binary Patterns)", "", TEAL)
bullets(s, Inches(0.6), Inches(6.0), Inches(5.9), Inches(1.3), [
    "Uniform LBP at radii 1/2/3 (P = 8/16/24) inside the brain mask \u2192 54 features/slice",
    "Mean + std pooled per view \u2192 576 texture features per patient",
], size=13.5)
box_label(s, Inches(6.9), Inches(1.5), Inches(5.9), Inches(0.9), "Clinical inputs", "", ORANGE)
bullets(s, Inches(6.9), Inches(2.55), Inches(5.9), Inches(2.0), [
    "Age \u00b7 sex \u00b7 Full-scale IQ (FIQ)",
    "Three explicit missingness flags; median imputation",
    "Only allowlisted fields \u2014 no site, ID, or diagnostic scores",
], size=13.5)
rect(s, Inches(6.9), Inches(4.95), Inches(5.9), Inches(2.0), LIGHT)
text(s, Inches(7.15), Inches(5.15), Inches(5.4), Inches(0.5), "Combined predictor", size=17, color=NAVY, bold=True)
text(s, Inches(7.15), Inches(5.7), Inches(5.4), Inches(0.9),
     "128 deep (PCA)  +  576 texture  +  6 clinical  =  710 features per patient",
     size=16, color=DARK)

# ---------------- Slide 8: Model ----------------
s = slide()
title_bar(s, "Classifier \u2014 Random Forest", "Ensemble of 500 decision trees")
bullets(s, Inches(0.6), Inches(1.5), Inches(6.6), Inches(5.4), [
    "500 trees, bootstrap sampling, class_weight = balanced_subsample.",
    "Hyperparameter search (8 candidates) over:",
    (1, "max depth, min samples per leaf, max features, max samples"),
    "5 outer \u00d7 3 inner cross-validation folds for model selection.",
    "Decision threshold tuned on inner out-of-fold probabilities.",
    "Ablations trained on identical folds: MRI-only, deep-only, texture-only, clinical-only.",
])
rect(s, Inches(7.6), Inches(1.5), Inches(5.2), Inches(5.3), LIGHT)
rect(s, Inches(7.6), Inches(1.5), Emu(38100), Inches(5.3), GREEN)
text(s, Inches(7.85), Inches(1.7), Inches(4.7), Inches(0.5), "Why Random Forest?", size=18, color=NAVY, bold=True)
bullets(s, Inches(7.85), Inches(2.3), Inches(4.7), Inches(4.4), [
    "Handles heterogeneous feature scales well",
    "Robust to overfitting on moderate cohorts",
    "Calibrated class probabilities \u2192 tunable threshold",
    "Feature-importance analysis is straightforward",
    "Well-suited to fusing deep + handcrafted + clinical blocks",
], size=13.5, gap=10)

# ---------------- Slide 9: Evaluation setup ----------------
s = slide()
title_bar(s, "Evaluation protocol", "How the model was measured")
box_label(s, Inches(0.6), Inches(1.6), Inches(5.8), Inches(1.0), "Patient-level split", "", NAVY2)
bullets(s, Inches(0.6), Inches(2.75), Inches(5.8), Inches(1.7), [
    "803 development / 201 test patients, stratified by site and label",
    "All slices of a patient stay together \u2014 evaluation is per person",
    "Test set evaluated once, after the models were frozen",
], size=13.5)
box_label(s, Inches(0.6), Inches(4.7), Inches(5.8), Inches(1.0), "Metrics", "", NAVY2)
bullets(s, Inches(0.6), Inches(5.85), Inches(5.8), Inches(1.3), [
    "Accuracy, sensitivity, specificity, balanced accuracy, precision, F1",
    "ROC-AUC, PR-AUC, confusion matrix, 95% confidence intervals",
], size=13.5)
rect(s, Inches(6.8), Inches(1.6), Inches(6.0), Inches(5.6), LIGHT)
rect(s, Inches(6.8), Inches(1.6), Emu(38100), Inches(5.6), TEAL)
text(s, Inches(7.05), Inches(1.8), Inches(5.5), Inches(0.5), "Reproducibility", size=18, color=NAVY, bold=True)
bullets(s, Inches(7.05), Inches(2.4), Inches(5.5), Inches(4.6), [
    "Fixed seed (42) \u2014 deterministic splits and models",
    "Frozen model bundles reproduce recorded predictions exactly",
    "Feature and weight fingerprints verify cache validity",
    "163 automated regression tests",
    "Full run artifacts and report preserved (runs/hybrid/r2)",
], size=13.5, gap=10)

# ---------------- Slide 10: Results ----------------
s = slide()
title_bar(s, "Results \u2014 held-out test set", "201 unseen patients \u00b7 primary combined model")
chip(s, Inches(0.6), Inches(1.45), Inches(3.0), Inches(1.15), "0.7164", "Accuracy  [0.651, 0.774]")
chip(s, Inches(3.75), Inches(1.45), Inches(3.0), Inches(1.15), "0.8183", "ROC-AUC  [0.760, 0.872]")
chip(s, Inches(6.9), Inches(1.45), Inches(3.0), Inches(1.15), "0.644 / 0.790", "Sensitivity / Specificity")
chip(s, Inches(10.05), Inches(1.45), Inches(3.0), Inches(1.15), "0.847", "PR-AUC")
s.shapes.add_picture(f"{FIG}/roc.png", Inches(0.7), Inches(2.95), height=Inches(4.1))
s.shapes.add_picture(f"{FIG}/pr.png", Inches(4.9), Inches(2.95), height=Inches(4.1))
s.shapes.add_picture(f"{FIG}/confusion_combined.png", Inches(9.2), Inches(2.95), height=Inches(3.4))
text(s, Inches(9.2), Inches(6.45), Inches(3.6), Inches(0.5),
     "TN 79 \u00b7 FP 21 \u00b7 FN 36 \u00b7 TP 65", size=14, color=GRAY, align=PP_ALIGN.CENTER)

# ---------------- Slide 11: Ablation analysis ----------------
s = slide()
title_bar(s, "Ablation analysis", "Contribution of each feature family (same folds)")
table(s, Inches(0.6), Inches(1.6), Inches(7.0), [
    ["Model", "Test accuracy", "ROC-AUC"],
    ["Combined (primary)", "0.716", "0.818"],
    ["MRI-only hybrid", "0.707", "0.790"],
    ["Texture only (GLCM+LBP)", "0.701", "0.777"],
    ["Deep only (EffNet+PCA)", "0.692", "0.765"],
    ["Clinical only (age/sex/FIQ)", "0.547", "0.578"],
], [3.4, 1.8, 1.8], size=14, row_h=0.42)
rect(s, Inches(8.0), Inches(1.6), Inches(4.8), Inches(5.4), LIGHT)
rect(s, Inches(8.0), Inches(1.6), Emu(38100), Inches(5.4), TEAL)
text(s, Inches(8.25), Inches(1.8), Inches(4.3), Inches(0.5), "Observations", size=18, color=NAVY, bold=True)
bullets(s, Inches(8.25), Inches(2.4), Inches(4.3), Inches(4.4), [
    "Combining all three sources gives the best result.",
    "Texture and deep features are complementary \u2014 each adds signal.",
    "Clinical inputs alone are weak; MRI carries the decision.",
    "B0 outperformed B5 \u2014 the larger model did not help at this cohort size.",
], size=13.5, gap=10)

# ---------------- Slide 12: Conclusion ----------------
s = slide()
title_bar(s, "Conclusion & future work", "")
bullets(s, Inches(0.6), Inches(1.6), Inches(6.4), Inches(5.4), [
    "Built a complete hybrid ASD classifier: EfficientNet embeddings + GLCM/LBP texture + clinical \u2192 Random Forest.",
    "Achieved 71.6% patient-level accuracy and 0.82 ROC-AUC on the held-out test set.",
    "The full pipeline \u2014 preprocessing, features, model, evaluation \u2014 is reproducible and fully tested.",
    "Future directions:",
    (1, "Supervised fine-tuning of the encoder on development patients"),
    (1, "3-D / volumetric encoders or medical-imaging pretrained backbones"),
    (1, "Gradient boosting / calibrated ensembling"),
    (1, "Larger multi-cohort data and external-site validation"),
], size=14.5, gap=8)
rect(s, Inches(7.4), Inches(1.6), Inches(5.4), Inches(5.3), LIGHT)
rect(s, Inches(7.4), Inches(1.6), Emu(38100), Inches(5.3), GREEN)
text(s, Inches(7.65), Inches(1.8), Inches(4.9), Inches(0.5), "Deliverables", size=18, color=NAVY, bold=True)
bullets(s, Inches(7.65), Inches(2.4), Inches(4.9), Inches(4.4), [
    "Reproducible training package (asd_training)",
    "Documented preprocessing + feature recipe",
    "Frozen model bundles with exact predictions",
    "Complete metrics, intervals, figures and report",
    "163 automated tests covering the full workflow",
], size=13.5, gap=10)
text(s, Inches(0.6), Inches(7.0), Inches(12), Inches(0.4),
     "Thank you \u2014 Questions?", size=20, color=NAVY, bold=True, align=PP_ALIGN.CENTER)

prs.save("ASD_MRI_Presentation_v2.pptx")
print("saved ASD_MRI_Presentation_v2.pptx with 12 slides")
