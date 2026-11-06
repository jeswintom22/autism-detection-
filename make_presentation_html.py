"""Generate a self-contained modern HTML slide deck (runs once)."""
import base64
import html
from pathlib import Path

FIG = Path("runs/hybrid/r2/figures")


def img64(name):
    return "data:image/png;base64," + base64.b64encode((FIG / name).read_bytes()).decode()


ROC, PR, CM = img64("roc.png"), img64("pr.png"), img64("confusion_combined.png")

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
  --bg: #0b1220; --bg2: #101a30; --card: rgba(255,255,255,.045); --line: rgba(255,255,255,.09);
  --txt: #e8eef7; --muted: #93a5c0; --teal: #2dd4bf; --cyan: #38bdf8; --orange: #fbbf24;
  --green: #34d399; --violet: #a78bfa;
}
html, body { height: 100%; background: var(--bg); color: var(--txt);
  font-family: 'Segoe UI', 'Inter', system-ui, sans-serif; overflow: hidden; }
body::before { content:''; position: fixed; inset: 0; z-index: -1;
  background:
    radial-gradient(60vw 60vh at 85% -10%, rgba(45,212,191,.14), transparent 60%),
    radial-gradient(50vw 50vh at -10% 90%, rgba(56,189,248,.12), transparent 60%),
    radial-gradient(40vw 40vh at 50% 110%, rgba(167,139,250,.10), transparent 60%); }
.slide { position: absolute; inset: 0; display: none; flex-direction: column;
  padding: 4.5rem 5.5rem 3.5rem; animation: fade .45s ease; }
.slide.active { display: flex; }
@keyframes fade { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
.eyebrow { font-size: .8rem; letter-spacing: .28em; text-transform: uppercase; color: var(--teal);
  font-weight: 700; margin-bottom: .6rem; }
h1 { font-size: 3.4rem; line-height: 1.08; font-weight: 800; letter-spacing: -.02em; }
h2 { font-size: 2.3rem; font-weight: 750; letter-spacing: -.01em; margin-bottom: 1.6rem; }
h2 span { background: linear-gradient(90deg, var(--teal), var(--cyan)); -webkit-background-clip: text;
  background-clip: text; color: transparent; }
.sub { color: var(--muted); font-size: 1.05rem; margin-top: -.9rem; margin-bottom: 1.4rem; }
.grad { background: linear-gradient(90deg, var(--teal), var(--cyan), var(--violet));
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.cols { display: flex; gap: 2.2rem; flex: 1; min-height: 0; }
.col { flex: 1; min-width: 0; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 16px;
  padding: 1.2rem 1.4rem; backdrop-filter: blur(6px); }
.card + .card { margin-top: 1rem; }
.card h3 { font-size: 1.05rem; font-weight: 700; margin-bottom: .55rem; display:flex; align-items:center; gap:.5rem; }
.tag { display: inline-block; width: .55rem; height: .55rem; border-radius: 50%; }
.teal { color: var(--teal); } .cyan { color: var(--cyan); } .orange { color: var(--orange); }
.green { color: var(--green); } .violet { color: var(--violet); }
ul { list-style: none; }
li { padding: .34rem 0 .34rem 1.5rem; position: relative; font-size: 1.02rem; color: #cbd7ea; line-height: 1.42; }
li::before { content: '\\25B8'; position: absolute; left: .2rem; color: var(--teal); }
li.sub { padding-left: 2.6rem; font-size: .93rem; color: var(--muted); }
li.sub::before { content: '\\2013'; color: var(--muted); }
.chips { display: flex; gap: 1.2rem; margin: 1rem 0 1.6rem; flex-wrap: wrap; }
.chip { flex: 1; min-width: 10rem; background: var(--card); border: 1px solid var(--line);
  border-radius: 16px; padding: 1.1rem 1.3rem; }
.chip .v { font-size: 1.9rem; font-weight: 800; letter-spacing: -.02em;
  background: linear-gradient(90deg, var(--teal), var(--cyan)); -webkit-background-clip: text;
  background-clip: text; color: transparent; }
.chip .l { color: var(--muted); font-size: .85rem; margin-top: .15rem; }
table { border-collapse: collapse; width: 100%; font-size: .98rem; }
th { text-align: left; color: var(--teal); font-weight: 700; font-size: .85rem;
  text-transform: uppercase; letter-spacing: .08em; padding: .55rem .8rem;
  border-bottom: 1px solid var(--line); }
td { padding: .5rem .8rem; border-bottom: 1px solid rgba(255,255,255,.05); color: #cbd7ea; }
tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: center; font-variant-numeric: tabular-nums; }
.flow { display: flex; align-items: stretch; gap: .8rem; flex: 1; }
.fstep { flex: 1; background: var(--card); border: 1px solid var(--line); border-radius: 16px;
  padding: 1.1rem 1rem; display: flex; flex-direction: column; justify-content: center; text-align: center; }
.fstep .t { font-weight: 700; font-size: 1.02rem; margin-bottom: .4rem; }
.fstep .d { color: var(--muted); font-size: .83rem; line-height: 1.4; }
.farrow { display: flex; align-items: center; color: var(--teal); font-size: 1.6rem; }
.figrow { display: flex; gap: 1.4rem; align-items: center; justify-content: center; flex: 1; }
.figrow img { height: 24rem; border-radius: 14px; background: #fff; padding: .5rem;
  border: 1px solid var(--line); }
.figrow img.cm { height: 20rem; }
.caption { text-align: center; color: var(--muted); font-size: .85rem; margin-top: .6rem; }
.nav { position: fixed; bottom: 1.4rem; right: 2rem; display: flex; gap: .55rem; z-index: 10; }
.nav button { background: var(--card); border: 1px solid var(--line); color: var(--txt);
  width: 2.6rem; height: 2.6rem; border-radius: 12px; font-size: 1.1rem; cursor: pointer;
  backdrop-filter: blur(6px); transition: .15s; }
.nav button:hover { background: rgba(45,212,191,.18); border-color: var(--teal); }
.counter { position: fixed; bottom: 1.75rem; left: 2rem; color: var(--muted); font-size: .9rem; z-index: 10; }
.progress { position: fixed; top: 0; left: 0; height: 3px; background: linear-gradient(90deg, var(--teal), var(--cyan));
  z-index: 10; transition: width .3s ease; }
.kbd { position: fixed; bottom: 1.75rem; right: 7.4rem; color: var(--muted); font-size: .78rem; z-index: 10; }
.badge { display: inline-block; padding: .18rem .7rem; border-radius: 99px; font-size: .8rem; font-weight: 700;
  border: 1px solid var(--line); background: rgba(45,212,191,.12); color: var(--teal); }
.slide.title { justify-content: center; align-items: flex-start; padding-left: 7rem; }
.slide.title h1 { font-size: 4rem; max-width: 62rem; }
.slide.title .sub2 { color: var(--muted); font-size: 1.25rem; margin-top: 1.6rem; line-height: 1.6; }
.hr { width: 7rem; height: 4px; border-radius: 4px; margin: 1.8rem 0;
  background: linear-gradient(90deg, var(--teal), var(--cyan)); }
"""

JS = """
const slides = [...document.querySelectorAll('.slide')];
const counter = document.getElementById('counter');
const bar = document.getElementById('progress');
let i = 0;
function show(n) {
  i = Math.max(0, Math.min(slides.length - 1, n));
  slides.forEach((s, k) => s.classList.toggle('active', k === i));
  counter.textContent = (i + 1) + ' / ' + slides.length;
  bar.style.width = ((i + 1) / slides.length * 100) + '%';
}
document.addEventListener('keydown', e => {
  if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); show(i + 1); }
  if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); show(i - 1); }
  if (e.key === 'Home') show(0);
  if (e.key === 'End') show(slides.length - 1);
});
document.getElementById('prev').onclick = () => show(i - 1);
document.getElementById('next').onclick = () => show(i + 1);
document.addEventListener('click', e => {
  if (e.target.closest('button') || e.target.closest('a')) return;
  show(i + 1);
});
show(0);
"""


def slide_html(eyebrow, title, body, subtitle=None):
    sub = f'<div class="sub">{subtitle}</div>' if subtitle else ''
    return f'''<section class="slide">
  <div class="eyebrow">{eyebrow}</div>
  <h2><span>{title}</span></h2>{sub}
  {body}
</section>'''


def bullets(items):
    out = ["<ul>"]
    for item in items:
        if isinstance(item, tuple):
            out.append(f'<li class="sub">{item[1]}</li>')
        else:
            out.append(f"<li>{item}</li>")
    out.append("</ul>")
    return "".join(out)


slides = []

# 1 — Title
slides.append('''<section class="slide title active">
  <div class="eyebrow">Mini Project &middot; 2026</div>
  <h1>Autism Spectrum Disorder Detection<br>from <span class="grad">Structural MRI</span></h1>
  <div class="hr"></div>
  <div class="sub2">A hybrid machine-learning pipeline<br>
  EfficientNet deep features &nbsp;+&nbsp; GLCM / LBP texture &nbsp;+&nbsp; clinical data
  &nbsp;&rarr;&nbsp; <b style="color:var(--teal)">Random Forest</b></div>
  <div class="sub2" style="margin-top:2.4rem; font-size:.95rem">
  ABIDE cohort &middot; 1,004 scans &middot; patient-level classification</div>
</section>''')

# 2 — Introduction
slides.append(slide_html("Introduction", "Why machine learning on structural MRI?", f'''
<div class="cols">
  <div class="col">{bullets([
    "Autism Spectrum Disorder (ASD) is a neurodevelopmental condition; early, objective detection supports timely intervention.",
    "Diagnosis today relies on behavioral assessment &mdash; subjective, time-consuming, and variable between clinicians.",
    "Structural MRI captures subtle neuroanatomical differences associated with ASD.",
    "Machine learning can learn patterns in MRI that are difficult to observe directly.",
    "<b>Goal:</b> an automated classifier that outputs P(ASD) for each scanned patient.",
  ])}</div>
  <div class="col"><div class="card">
    <h3><span class="tag" style="background:var(--teal)"></span>Project objectives</h3>
    {bullets([
      "Combine deep and handcrafted image features",
      "Add approved clinical inputs (age, sex, FIQ)",
      "Patient-level classification &mdash; one decision per person",
      "Rigorous, reproducible evaluation protocol",
    ])}
  </div></div>
</div>'''))

# 3 — Dataset
slides.append(slide_html("Dataset", "ABIDE structural MRI", f'''
<div class="chips">
  <div class="chip"><div class="v">1,004</div><div class="l">skull-stripped T1 scans</div></div>
  <div class="chip"><div class="v">504 / 500</div><div class="l">control / ASD</div></div>
  <div class="chip"><div class="v">20+</div><div class="l">acquisition sites</div></div>
  <div class="chip"><div class="v">803 / 201</div><div class="l">development / test</div></div>
</div>
<div class="cols">
  <div class="col" style="flex:1.4"><div class="card"><table>
    <tr><th>Property</th><th>Details</th></tr>
    <tr><td>Modality</td><td>Skull-stripped T1-weighted structural MRI (.nii.gz)</td></tr>
    <tr><td>Classes</td><td>Control (0) vs ASD (1) &mdash; nearly balanced</td></tr>
    <tr><td>Phenotype inputs used</td><td>Age, sex, FIQ (allowlisted; no diagnostic scores)</td></tr>
    <tr><td>Acquisition</td><td>Multi-site &mdash; mixed orientations and voxel spacings</td></tr>
    <tr><td>Split</td><td>80 / 20 patient-level, stratified by site and label</td></tr>
  </table></div></div>
  <div class="col"><div class="card">
    <h3><span class="tag" style="background:var(--orange)"></span>Data quality checks</h3>
    {bullets([
      "All IDs verified against the phenotype table",
      "Valid 3-D geometry confirmed for every scan",
      "File- and voxel-level fingerprints recorded",
      "Missing clinical values flagged explicitly",
    ])}
  </div></div>
</div>''', "Skull-stripped T1-weighted scans"))

# 4 — Method overview
slides.append(slide_html("Method overview", "Raw MRI &rarr; one decision per patient", '''
<div class="flow" style="margin-top:1rem">
  <div class="fstep"><div class="t cyan">Skull-stripped<br>T1 MRI</div><div class="d">.nii.gz<br>mixed spacings</div></div>
  <div class="farrow">&rarr;</div>
  <div class="fstep"><div class="t teal">Preprocessing</div><div class="d">RAS &middot; 1 mm iso &middot;<br>intensity norm &middot;<br>9 slices &times; 3 views</div></div>
  <div class="farrow">&rarr;</div>
  <div class="fstep"><div class="t">EfficientNet B0/B5</div><div class="d">ImageNet embeddings<br>per slice &rarr; pooled</div></div>
  <div class="fstep"><div class="t teal">GLCM + LBP</div><div class="d">energy, homogeneity,<br>contrast, entropy &hellip;</div></div>
  <div class="fstep"><div class="t orange">Clinical</div><div class="d">age &middot; sex &middot; FIQ</div></div>
  <div class="farrow">&rarr;</div>
  <div class="fstep"><div class="t violet">Random Forest</div><div class="d">500 trees &middot; search &middot;<br>threshold tuning</div></div>
  <div class="farrow">&rarr;</div>
  <div class="fstep" style="border-color:var(--green)"><div class="t green">P(ASD)</div><div class="d">per patient</div></div>
</div>
<div class="caption" style="font-size:1rem; margin-top:2rem">Three complementary views of each patient are fused into a single classifier.</div>'''))

# 5 — Preprocessing
slides.append(slide_html("Preprocessing", "Standardizing every scan", f'''
<div class="cols">
  <div class="col" style="flex:1.5"><div class="card"><table>
    <tr><th>Step</th><th>Specification</th></tr>
    <tr><td>Orientation</td><td>Canonicalize every scan to RAS</td></tr>
    <tr><td>Resampling</td><td>Axis-aligned 1 mm isotropic grid (linear image / nearest mask)</td></tr>
    <tr><td>Brain mask</td><td>Foreground from skull-stripped voxels, re-applied after resampling</td></tr>
    <tr><td>Normalization</td><td>Per-patient in-mask 1st&ndash;99th percentile &rarr; clip [0, 1]</td></tr>
    <tr><td>Slice selection</td><td>9 slices &times; axial / coronal / sagittal, central 20&ndash;80% extent</td></tr>
    <tr><td>Quality gates</td><td>&ge; 256 brain pixels/slice; failures reported explicitly</td></tr>
    <tr><td>CNN input</td><td>Aspect-preserving letterbox &rarr; 224 (B0) / 456 (B5)</td></tr>
  </table></div></div>
  <div class="col"><div class="card">
    <h3><span class="tag" style="background:var(--teal)"></span>Design choices</h3>
    {bullets([
      "Fresh processing from original scans &mdash; no stale caches",
      "Slices sampled across axial, coronal and sagittal anatomy",
      "Intensity scale normalized inside the brain mask only",
      "Deterministic recipe &mdash; identical pipeline for every scan",
    ])}
  </div></div>
</div>'''))

# 6 — Deep features
slides.append(slide_html("Deep features", "EfficientNet embeddings", f'''
<div class="cols">
  <div class="col" style="flex:1.3">{bullets([
    "Pretrained EfficientNet as a fixed feature extractor (B0 first, B5 compared).",
    "Classifier head removed &rarr; global convolutional embedding per slice:",
    ("sub", "1,280 dimensions (B0) &nbsp;&middot;&nbsp; 2,048 dimensions (B5)"),
    "27 slices per patient (9 &times; 3 views) &rarr; mean + std pooled per view:",
    ("sub", "7,680 (B0) / 12,288 (B5) deep features per patient"),
    "Fold-local standardization + PCA &rarr; 128 compact deep features.",
    "Grayscale replicated to 3 channels, ImageNet mean/std normalization.",
  ])}</div>
  <div class="col"><div class="card">
    <h3><span class="tag" style="background:var(--cyan)"></span>Why EfficientNet?</h3>
    {bullets([
      "Compound-scaled architecture &mdash; strong accuracy per parameter",
      "ImageNet features transfer well as generic image descriptors",
      "Frozen encoder &rarr; reproducible, fast, no GPU-heavy training",
      "Two sizes compared fairly on the same folds",
    ])}
  </div></div>
</div>''', "Transfer learning from ImageNet"))

# 7 — Texture + clinical
slides.append(slide_html("Handcrafted + clinical", "Interpretable descriptors", f'''
<div class="cols">
  <div class="col">
    <div class="card">
      <h3><span class="tag" style="background:var(--teal)"></span>GLCM texture</h3>
      {bullets([
        "Contrast &middot; dissimilarity &middot; homogeneity &middot; ASM &middot; energy &middot; correlation &middot; entropy",
        "32 levels, distances 1/2/4, four angles, brain-mask pairs only &rarr; <b>42 / slice</b>",
      ])}
    </div>
    <div class="card">
      <h3><span class="tag" style="background:var(--teal)"></span>LBP texture</h3>
      {bullets([
        "Uniform LBP at radii 1/2/3 (P = 8/16/24), in-mask centers &rarr; <b>54 / slice</b>",
        "Mean + std pooled per view &rarr; <b>576 per patient</b>",
      ])}
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3><span class="tag" style="background:var(--orange)"></span>Clinical inputs</h3>
      {bullets([
        "Age &middot; sex &middot; Full-scale IQ (FIQ)",
        "Three explicit missingness flags; median imputation",
        "Allowlisted fields only &mdash; no site, ID, or diagnostic scores",
      ])}
    </div>
    <div class="card">
      <h3><span class="tag" style="background:var(--violet)"></span>Combined predictor</h3>
      <div style="font-size:1.2rem; line-height:1.6">
        <b class="cyan">128</b> deep (PCA) &nbsp;+&nbsp; <b class="teal">576</b> texture
        &nbsp;+&nbsp; <b class="orange">6</b> clinical &nbsp;=&nbsp; <b class="green">710</b> features per patient
      </div>
    </div>
  </div>
</div>'''))

# 8 — Model
slides.append(slide_html("Classifier", "Random Forest", f'''
<div class="cols">
  <div class="col" style="flex:1.3">{bullets([
    "500 trees, bootstrap sampling, class_weight = balanced_subsample.",
    "Hyperparameter search (8 candidates) over:",
    ("sub", "max depth &middot; min samples per leaf &middot; max features &middot; max samples"),
    "5 outer &times; 3 inner cross-validation folds for model selection.",
    "Decision threshold tuned on inner out-of-fold probabilities.",
    "Ablations trained on identical folds: MRI-only, deep-only, texture-only, clinical-only.",
  ])}</div>
  <div class="col"><div class="card">
    <h3><span class="tag" style="background:var(--green)"></span>Why Random Forest?</h3>
    {bullets([
      "Handles heterogeneous feature scales well",
      "Robust to overfitting on moderate cohorts",
      "Calibrated class probabilities &rarr; tunable threshold",
      "Feature-importance analysis is straightforward",
      "Well-suited to fusing deep + handcrafted + clinical blocks",
    ])}
  </div></div>
</div>''', "Ensemble of 500 decision trees"))

# 9 — Evaluation setup
slides.append(slide_html("Evaluation", "How the model was measured", f'''
<div class="cols">
  <div class="col">
    <div class="card">
      <h3><span class="tag" style="background:var(--cyan)"></span>Patient-level split</h3>
      {bullets([
        "803 development / 201 test patients, stratified by site and label",
        "All slices of a patient stay together &mdash; evaluation is per person",
        "Test set evaluated once, after the models were frozen",
      ])}
    </div>
    <div class="card">
      <h3><span class="tag" style="background:var(--cyan)"></span>Metrics</h3>
      {bullets([
        "Accuracy, sensitivity, specificity, balanced accuracy, precision, F1",
        "ROC-AUC, PR-AUC, confusion matrix, 95% confidence intervals",
      ])}
    </div>
  </div>
  <div class="col"><div class="card">
    <h3><span class="tag" style="background:var(--teal)"></span>Reproducibility</h3>
    {bullets([
      "Fixed seed (42) &mdash; deterministic splits and models",
      "Frozen model bundles reproduce recorded predictions exactly",
      "Feature and weight fingerprints verify cache validity",
      "163 automated regression tests",
      "Full run artifacts and report preserved (runs/hybrid/r2)",
    ])}
  </div></div>
</div>'''))

# 10 — Results
slides.append(slide_html("Results", "Held-out test set", f'''
<div class="chips">
  <div class="chip"><div class="v">0.7164</div><div class="l">Accuracy &nbsp;[0.651, 0.774]</div></div>
  <div class="chip"><div class="v">0.8183</div><div class="l">ROC-AUC &nbsp;[0.760, 0.872]</div></div>
  <div class="chip"><div class="v">0.644 / 0.790</div><div class="l">Sensitivity / Specificity</div></div>
  <div class="chip"><div class="v">0.847</div><div class="l">PR-AUC</div></div>
</div>
<div class="figrow">
  <div><img src="{ROC}" alt="ROC"><div class="caption">ROC curves &mdash; all model families</div></div>
  <div><img src="{PR}" alt="PR"><div class="caption">Precision&ndash;recall curves</div></div>
  <div><img class="cm" src="{CM}" alt="Confusion matrix"><div class="caption">Confusion matrix &mdash; TN 79 &middot; FP 21 &middot; FN 36 &middot; TP 65</div></div>
</div>''', "201 unseen patients &middot; primary combined model"))

# 11 — Ablation
slides.append(slide_html("Ablation analysis", "Contribution of each feature family", f'''
<div class="cols">
  <div class="col" style="flex:1.3"><div class="card"><table>
    <tr><th>Model</th><th class="num">Test accuracy</th><th class="num">ROC-AUC</th></tr>
    <tr><td><b>Combined (primary)</b></td><td class="num"><b class="teal">0.716</b></td><td class="num"><b class="teal">0.818</b></td></tr>
    <tr><td>MRI-only hybrid</td><td class="num">0.707</td><td class="num">0.790</td></tr>
    <tr><td>Texture only (GLCM+LBP)</td><td class="num">0.701</td><td class="num">0.777</td></tr>
    <tr><td>Deep only (EffNet+PCA)</td><td class="num">0.692</td><td class="num">0.765</td></tr>
    <tr><td>Clinical only (age/sex/FIQ)</td><td class="num">0.547</td><td class="num">0.578</td></tr>
  </table></div></div>
  <div class="col"><div class="card">
    <h3><span class="tag" style="background:var(--teal)"></span>Observations</h3>
    {bullets([
      "Combining all three sources gives the best result.",
      "Texture and deep features are complementary &mdash; each adds signal.",
      "Clinical inputs alone are weak; MRI carries the decision.",
      "B0 outperformed B5 &mdash; the larger model did not help at this cohort size.",
    ])}
  </div></div>
</div>''', "Same folds for a fair comparison"))

# 12 — Conclusion
slides.append(slide_html("Conclusion", "Summary &amp; future work", f'''
<div class="cols">
  <div class="col" style="flex:1.25">{bullets([
    "Built a complete hybrid ASD classifier: EfficientNet embeddings + GLCM/LBP texture + clinical &rarr; Random Forest.",
    "Achieved <b>71.6% patient-level accuracy</b> and <b>0.82 ROC-AUC</b> on the held-out test set.",
    "The full pipeline &mdash; preprocessing, features, model, evaluation &mdash; is reproducible and fully tested.",
    "<b>Future directions:</b>",
    ("sub", "Supervised fine-tuning of the encoder on development patients"),
    ("sub", "3-D / volumetric encoders or medical-imaging pretrained backbones"),
    ("sub", "Gradient boosting / calibrated ensembling"),
    ("sub", "Larger multi-cohort data and external-site validation"),
  ])}</div>
  <div class="col"><div class="card">
    <h3><span class="tag" style="background:var(--green)"></span>Deliverables</h3>
    {bullets([
      "Reproducible training package (asd_training)",
      "Documented preprocessing + feature recipe",
      "Frozen model bundles with exact predictions",
      "Complete metrics, intervals, figures and report",
      "163 automated tests covering the full workflow",
    ])}
  </div></div>
</div>
<div style="text-align:center; margin-top:2rem; font-size:1.5rem; font-weight:700">
  Thank you &mdash; <span class="grad">Questions?</span></div>'''))

doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASD Detection from Structural MRI</title>
<style>{CSS}</style></head>
<body>
<div class="progress" id="progress"></div>
{''.join(slides)}
<div class="counter" id="counter"></div>
<div class="kbd">&larr; &rarr; navigate &middot; click advances</div>
<div class="nav"><button id="prev">&#8249;</button><button id="next">&#8250;</button></div>
<script>{JS}</script>
</body></html>"""

Path("ASD_MRI_Presentation.html").write_text(doc, encoding="utf-8")
size_kb = Path("ASD_MRI_Presentation.html").stat().st_size // 1024
print(f"saved ASD_MRI_Presentation.html ({size_kb} KB, {len(slides)} slides)")
