import math
import re
from pathlib import Path

from . import artifacts

METRIC_ROWS = (
    ("accuracy", "Accuracy", ("accuracy",)),
    ("balanced_accuracy", "Balanced accuracy", ("balanced_accuracy",)),
    ("sensitivity", "Sensitivity (ASD recall)", ("sensitivity", "recall")),
    ("specificity", "Specificity (control)", ("specificity",)),
    ("precision", "Precision", ("precision",)),
    ("f1", "F1", ("f1",)),
    ("roc_auc", "ROC-AUC", ("roc_auc",)),
    ("average_precision", "PR-AUC (average precision)", ("average_precision", "pr_auc")),
)
FAMILY_LABELS = {
    "primary": "Primary combined (EfficientNet + GLCM/LBP + saliency)",
    "combined": "Primary combined (EfficientNet + GLCM/LBP + saliency)",
    "mri_hybrid": "MRI-only hybrid (EfficientNet + GLCM/LBP)",
    "mri_only": "MRI-only hybrid (EfficientNet + GLCM/LBP)",
    "deep_only": "Deep-only (EfficientNet + PCA)",
    "texture_only": "Texture-only (GLCM + LBP)",
    "saliency_only": "Saliency-only",
}
MISSING = "not available"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def _fmt(value):
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return MISSING if not math.isfinite(value) else f"{value:.4f}"
    return str(value).replace("|", "/")


def _first(mapping, keys):
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _interval_text(interval):
    if interval is None:
        return MISSING
    if isinstance(interval, dict):
        lo = _first(interval, ("low", "lower", "ci_low", "lo"))
        hi = _first(interval, ("high", "upper", "ci_high", "hi"))
        if lo is None and hi is None:
            return MISSING
        return f"[{_fmt(lo)}, {_fmt(hi)}]"
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        return f"[{_fmt(interval[0])}, {_fmt(interval[1])}]"
    return MISSING


def _metric_interval(intervals, key, aliases):
    if not isinstance(intervals, dict):
        return None
    for name in (key,) + tuple(aliases):
        if name in intervals:
            return intervals[name]
    return None


def _table(rows, columns):
    if not rows:
        return "_No rows supplied._\n"
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |")
    return "\n".join(lines) + "\n"


def _generic_table(rows):
    if not rows:
        return "_No rows supplied._\n"
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return _table(rows, [(key, key) for key in keys])


def _confusion(metrics):
    matrix = _first(metrics, ("confusion_matrix", "confusion"))
    if isinstance(matrix, dict):
        keys = ("tn", "fp", "fn", "tp")
        if all(key in matrix for key in keys):
            return [[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]]
        return None
    if isinstance(matrix, (list, tuple)) and len(matrix) == 2 and all(len(r) == 2 for r in matrix):
        return [list(r) for r in matrix]
    if all(key in metrics for key in ("tn", "fp", "fn", "tp")):
        return [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    return None


def _plot_curves(models, test_curves, figures_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for kind, x_key, y_key, xlabel, ylabel in (
        ("roc", "fpr", "tpr", "False positive rate", "True positive rate"),
        ("pr", "recall", "precision", "Recall", "Precision"),
    ):
        fig, ax = plt.subplots(figsize=(5, 5))
        drawn = False
        for family, metrics in models.items():
            family_curves = metrics.get("curves") or (test_curves or {}).get(family)
            curve = _first(family_curves if isinstance(family_curves, dict) else None, (kind,))
            if not isinstance(curve, dict) or x_key not in curve or y_key not in curve:
                continue
            ax.plot(curve[x_key], curve[y_key], label=FAMILY_LABELS.get(family, family))
            drawn = True
        if drawn:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(kind.upper() + " (stored coordinates)")
            ax.legend(fontsize=6)
            path = figures_dir / f"{kind}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
    return written


def _plot_confusions(models, figures_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for family, metrics in models.items():
        matrix = _confusion(metrics)
        if matrix is None:
            continue
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
        ax.set_xticks([0, 1], ["control", "ASD"])
        ax.set_yticks([0, 1], ["control", "ASD"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(family, fontsize=8)
        path = figures_dir / f"confusion_{family}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def _test_section(test, figures):
    lines = ["## Locked patient-test evaluation", ""]
    if not test:
        lines += [
            "**Incomplete: no test evaluation has been recorded.** The 80% target has not been evaluated.",
            "",
        ]
        return lines
    models = test.get("models") or {}
    primary = test.get("primary_model")
    intervals = test.get("intervals") or {}
    lines.append(f"Test patients: {_fmt(test.get('n_patients'))}. Designated primary model: {_fmt(primary)}.")
    target = test.get("target_met")
    if target is None:
        lines.append("")
        lines.append("**Incomplete: target_met was not recorded; the 80% target status is unknown.**")
    else:
        verdict = "MET" if target else "NOT MET"
        lines.append("")
        lines.append(f"**Target (unrounded patient accuracy > 0.80) {verdict} by the primary model.**")
    lines.append("")
    if not models:
        lines += ["**Incomplete: no model metrics were recorded.**", ""]
        return lines
    for family, metrics in models.items():
        label = FAMILY_LABELS.get(family, family)
        lines += [f"### {label}" + (" (primary)" if family == primary else ""), ""]
        family_intervals = intervals.get(family) or {}
        rows = []
        for key, name, aliases in METRIC_ROWS:
            rows.append({
                "metric": name,
                "value": _first(metrics, (key,) + aliases),
                "interval": _interval_text(_metric_interval(family_intervals, key, aliases)),
            })
        lines.append(_table(rows, [("metric", "Metric"), ("value", "Value"), ("interval", "95% interval")]))
        matrix = _confusion(metrics)
        if matrix is None:
            lines += ["Confusion matrix: not available.", ""]
        else:
            lines += [
                "Confusion matrix (rows = true control/ASD, columns = predicted control/ASD):",
                "",
                "| | pred control | pred ASD |",
                "| --- | --- | --- |",
                f"| true control | {matrix[0][0]} | {matrix[0][1]} |",
                f"| true ASD | {matrix[1][0]} | {matrix[1][1]} |",
                "",
            ]
        threshold = _first(metrics, ("threshold",))
        if threshold is not None:
            lines += [f"Frozen threshold: {_fmt(threshold)}", ""]
    subgroups = test.get("subgroups")
    lines += ["### Subgroups (descriptive only)", ""]
    if not subgroups:
        lines += ["Subgroup results: not available.", ""]
    else:
        for name, content in subgroups.items():
            lines += [f"**{name}**", ""]
            if isinstance(content, list):
                lines.append(_generic_table(content))
            elif isinstance(content, dict) and all(isinstance(v, dict) for v in content.values()):
                lines.append(_generic_table([{"group": k, **v} for k, v in content.items()]))
            else:
                lines += [f"{_fmt(content)}", ""]
    if figures:
        lines += ["### Figures", ""]
        lines += [f"![{p.stem}](figures/{p.name})" for p in figures]
        lines.append("")
    else:
        lines += ["ROC/PR curve coordinates were not stored; no curve plots were drawn.", ""]
    return lines


def render_report(run_dir):
    run_dir = Path(run_dir)
    artifacts.ensure_allowed_output(run_dir)
    data_path = run_dir / "report_data.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"{data_path} not found; nothing to report")
    data = artifacts.read_json(data_path)
    status = data.get("status")
    development = data.get("development") or {}
    test = data.get("test")

    figures = []
    if test and isinstance(test.get("models"), dict):
        unsafe = [name for name in test["models"] if not SAFE_NAME.fullmatch(str(name))]
        if unsafe:
            raise ValueError(f"unsafe model family name(s) in report_data.json: {unsafe}")
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        figures = _plot_curves(test["models"], test.get("curves"), figures_dir) + _plot_confusions(test["models"], figures_dir)

    lines = ["# Patient-safe EfficientNet + texture + Random Forest report", ""]
    lines += [f"Status: **{_fmt(status)}**", f"Primary model: {_fmt(data.get('primary_model'))}", ""]
    if status not in ("complete", "completed", "final"):
        lines += ["**This report is incomplete: not every stage has produced saved evidence.**", ""]
    lines += [
        "## Scope and caveats",
        "",
        "- The primary model fuses MRI deep features (EfficientNet conv-map), handcrafted texture "
        "(GLCM + LBP), and saliency-map statistics. No clinical variables are used.",
        "- The 20% test set is an internal, historically explored holdout, not a never-seen external cohort.",
        "- The >80% held-out accuracy target is a goal and is not guaranteed; the result below is whatever was measured.",
        "- No unseen-site generalization guarantee and no clinical diagnostic validation.",
        "- Development results are model-selection results and are not the final test claim.",
        "",
        "## Development comparison",
        "",
        "### Backbone comparison",
        "",
        _generic_table(development.get("backbone_comparison") or []),
        "### Ablation comparison (fixed settings, not independently optimized)",
        "",
        _generic_table(development.get("ablation_comparison") or []),
    ]
    if not development.get("backbone_comparison"):
        lines += ["**Incomplete: backbone comparison missing.**", ""]
    if not development.get("ablation_comparison"):
        lines += ["**Incomplete: ablation comparison missing.**", ""]
    lines += _test_section(test, figures)
    limitations = data.get("limitations") or []
    lines += ["## Limitations", ""]
    if limitations:
        lines += [f"- {item}" for item in limitations]
    else:
        lines.append("No limitations were recorded in report_data.json.")
    lines.append("")

    output = run_dir / "report.md"
    artifacts.atomic_text(output, "\n".join(lines))
    return output
