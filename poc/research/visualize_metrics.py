#!/usr/bin/env python3
"""Visualize augmentation metrics. Separate charts for model cards and datasheets.

Usage:
    pip install matplotlib
    python poc/visualize_metrics.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

CSV_PATH = Path(__file__).parent / "data" / "outputs" / "metrics_comparison.csv"
JUDGE_CSV = Path(__file__).parent / "data" / "outputs" / "judge_scores.csv"
CHART_DIR = Path(__file__).parent / "data" / "outputs" / "charts"

MC_FIELDS = [
    "name", "version", "short_description", "full_description", "keywords", "author",
    "citation", "input_data", "input_type", "output_data", "foundational_model",
    "category", "documentation", "is_private", "is_gated", "ai_model_framework",
    "ai_model_license", "ai_model_model_type", "ai_model_version", "ai_model_description",
    "ai_model_owner", "ai_model_location",
]
DS_FIELDS = [
    "title", "description", "subjects", "creator", "publisher", "resource_type",
    "resource_type_general", "publication_year", "size", "format", "version", "license",
]
MC_REQUIRED = {"name", "category", "input_type", "keywords", "author", "short_description", "ai_model_license", "ai_model_framework"}
DS_REQUIRED = {"title", "description", "subjects", "resource_type_general", "creator"}

# MLHub ModelMetadata fields (36) — Category A (9) + Category B (6) + Category C (21)
MLHUB_FIELDS = [
    "name", "author", "model_type", "license", "task_types",
    "inference_software_dependencies", "pretraining_datasets", "keywords", "libraries",
    "inference_hardware", "inference_max_energy_consumption_watts",
    "inference_max_latency_ms", "inference_min_throughput",
    "inference_max_memory_usage_mb", "inference_distributed",
    "image", "multi_modal", "model_inputs", "model_outputs",
    "inference_precision", "inference_max_compute_utilization_percentage",
    "pretrained", "edge_optimized", "quantization_aware", "supports_quantization",
    "pruned", "slimmed",
    "training_time", "training_precision", "training_hardware",
    "training_max_energy_consumption_watts", "training_distributed",
    "finetuning_datasets", "regulatory", "bias_evaluation_score", "annotations",
]
MLHUB_REQUIRED = {"name", "author", "task_types", "license",
                  "inference_hardware", "inference_max_latency_ms"}


def load_csv():
    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            for k in ("attr_confidence", "composite", "exact_pct", "semantic_overlap", "coverage_pct"):
                row[k] = float(row[k])
            rows.append(row)
    return rows


def load_judge_scores():
    if not JUDGE_CSV.exists():
        return {}
    scores: dict[tuple[str, str], list[float]] = {}
    with open(JUDGE_CSV) as f:
        for row in csv.DictReader(f):
            key = (row["phase"], row["field"])
            s = float(row["judge_score"])
            if s >= 0:
                scores.setdefault(key, []).append(s / 2.0)
    return {k: sum(v) / len(v) for k, v in scores.items()}


def pivot(rows):
    data = {}
    for row in rows:
        data[(row["phase"], row["field"])] = row
    return data


def _get(data, phase, field, metric, default=0.0):
    row = data.get((phase, field))
    return row[metric] if row else default


def _record_metrics(data, phase, fields, required):
    total = len(fields)
    filled = sum(1 for f in fields if _get(data, phase, f, "coverage_pct") > 0)
    req_filled = sum(1 for f in fields if f in required and _get(data, phase, f, "coverage_pct") > 0)
    conf_sum = sum(_get(data, phase, f, "attr_confidence") for f in fields)
    return filled / total, req_filled / len(required), conf_sum / total


def _has_data(data, phases, fields):
    return any(_get(data, p, f, "coverage_pct") > 0 or _get(data, p, f, "attr_confidence") > 0
               for p in phases for f in fields)


def chart_metric_overview(data, phases, fields, required, label, filename, judge_scores,
                          phase_field_map: dict | None = None,
                          display_names: dict | None = None):
    """Render the metric-overview heatmap.

    If `phase_field_map` is provided, it overrides (fields, required) per phase —
    used to mix M1/M2/M3 (Patra MC_FIELDS) with M4: Patra→MLHub (MLHUB_FIELDS)
    in the same chart.

    `display_names` remaps phase keys to axis labels (e.g. "M4: Patra→MLHub" → "Patra → MLHub").
    """
    row_labels = []
    matrix = []
    for phase in phases:
        row_fields, row_required = fields, required
        if phase_field_map and phase in phase_field_map:
            row_fields, row_required = phase_field_map[phase]
        c, s, o = _record_metrics(data, phase, row_fields, row_required)
        judge_vals = [judge_scores.get((phase, f), -1) for f in row_fields if _get(data, phase, f, "coverage_pct") > 0]
        judge_valid = [v for v in judge_vals if v >= 0]
        accuracy = sum(judge_valid) / len(judge_valid) if judge_valid else -1.0
        axis_label = (display_names or {}).get(phase, phase)
        row_labels.append(axis_label)
        matrix.append([c, s, o, accuracy])

    matrix = np.array(matrix)
    col_labels = ["Completeness", "Sufficiency", "Overall\nConfidence", "Accuracy\n(Judge)"]

    fig, ax = plt.subplots(figsize=(8, max(2.5, len(row_labels) * 0.7 + 1)))
    im = ax.imshow(matrix, cmap=plt.cm.RdYlGn, norm=mcolors.Normalize(vmin=0.0, vmax=1.0), aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.xaxis.set_ticks_position("top")
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if val < 0:
                ax.text(j, i, "—", ha="center", va="center", fontsize=12, color="gray")
            else:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11,
                        fontweight="bold", color="white" if val < 0.4 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Score")
    ax.set_title(f"All Metrics — {label}", fontsize=12, fontweight="bold", pad=40)
    fig.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_attr_confidence(data, phases, fields, label, filename):
    field_labels = [f.replace("ai_model_", "ai.") for f in fields]
    matrix = np.array([[_get(data, p, f, "attr_confidence") for p in phases] for f in fields])

    fig, ax = plt.subplots(figsize=(max(4, len(phases) * 2.5 + 1), max(5, len(fields) * 0.35 + 1.5)))
    im = ax.imshow(matrix, cmap=plt.cm.RdYlGn, norm=mcolors.Normalize(vmin=0.0, vmax=1.0), aspect="auto")
    ax.set_yticks(range(len(field_labels)))
    ax.set_yticklabels(field_labels, fontsize=8)
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phases, fontsize=9, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    for i in range(len(field_labels)):
        for j in range(len(phases)):
            val = matrix[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if val < 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Attribute Confidence", location="bottom", pad=0.08)
    ax.set_title(f"Attribute Confidence — {label}", fontsize=12, fontweight="bold", pad=25)
    fig.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_coverage(data, phases, fields, label, filename):
    field_labels = [f.replace("ai_model_", "ai.") for f in fields]
    matrix = np.array([[_get(data, p, f, "coverage_pct") / 100.0 for p in phases] for f in fields])

    fig, ax = plt.subplots(figsize=(max(4, len(phases) * 2.5 + 1), max(5, len(fields) * 0.35 + 1.5)))
    im = ax.imshow(matrix, cmap=plt.cm.Blues, norm=mcolors.Normalize(vmin=0.0, vmax=1.0), aspect="auto")
    ax.set_yticks(range(len(field_labels)))
    ax.set_yticklabels(field_labels, fontsize=8)
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phases, fontsize=9, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    for i in range(len(field_labels)):
        for j in range(len(phases)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if val < 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Coverage", location="bottom", pad=0.08)
    ax.set_title(f"Field Coverage — {label}", fontsize=12, fontweight="bold", pad=25)
    fig.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_conf_vs_coverage(data, phases, fields, label, filename):
    color_cycle = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db", "#9b59b6"]
    phase_colors = {p: color_cycle[i % len(color_cycle)] for i, p in enumerate(phases)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for phase in phases:
        xs, ys, labels = [], [], []
        for f in fields:
            ac = _get(data, phase, f, "attr_confidence")
            cov = _get(data, phase, f, "coverage_pct") / 100.0
            xs.append(ac)
            ys.append(cov)
            labels.append(f)
        ax.scatter(xs, ys, label=phase, color=phase_colors[phase], s=60, alpha=0.8,
                   edgecolors="white", linewidth=0.5)
        for x, y, lab in zip(xs, ys, labels):
            if y < 0.5 or x < 0.5:
                ax.annotate(lab.replace("ai_model_", "ai.").replace("_", " "),
                            (x, y), textcoords="offset points", xytext=(6, 4),
                            fontsize=6, alpha=0.8, color=phase_colors[phase])

    ax.set_xlabel("Attribute Confidence", fontsize=10)
    ax.set_ylabel("Coverage (% filled)", fontsize=10)
    ax.set_title(f"Confidence vs Coverage — {label}", fontweight="bold", fontsize=12)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.2)
    ax.axvline(x=0.7, color="gray", linestyle="--", alpha=0.2)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_attr_and_coverage(data, phases, fields, label, filename, display_names: dict | None = None):
    """Combined heatmap: attr_confidence (left) + coverage (right), shared field axis."""
    field_labels = [f.replace("_", " ") for f in fields]
    conf_matrix = np.array([[_get(data, p, f, "attr_confidence") for p in phases] for f in fields])
    cov_matrix = np.array([[_get(data, p, f, "coverage_pct") / 100.0 for p in phases] for f in fields])
    axis_labels = [(display_names or {}).get(p, p) for p in phases]

    fig, (ax_conf, ax_cov) = plt.subplots(
        1, 2,
        figsize=(max(6, len(phases) * 4 + 2), max(7, len(fields) * 0.35 + 1.5)),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    # Left panel — attribute confidence (RdYlGn 0→1)
    im_c = ax_conf.imshow(conf_matrix, cmap=plt.cm.RdYlGn,
                          norm=mcolors.Normalize(vmin=0.0, vmax=1.0), aspect="auto")
    ax_conf.set_yticks(range(len(field_labels)))
    ax_conf.set_yticklabels(field_labels, fontsize=8)
    ax_conf.set_xticks(range(len(phases)))
    ax_conf.set_xticklabels(axis_labels, fontsize=9, fontweight="bold")
    ax_conf.xaxis.set_ticks_position("top")
    ax_conf.set_title("Attribute Confidence", fontsize=11, fontweight="bold", pad=25)
    for i in range(len(field_labels)):
        for j in range(len(phases)):
            val = conf_matrix[i, j]
            if val > 0:
                ax_conf.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                             color="white" if val < 0.5 else "black")
    # fig.colorbar(im_c, ax=ax_conf, shrink=0.7, location="bottom", pad=0.06)

    # Right panel — field coverage (Blues 0→1)
    im_v = ax_cov.imshow(cov_matrix, cmap=plt.cm.Blues,
                         norm=mcolors.Normalize(vmin=0.0, vmax=1.0), aspect="auto")
    ax_cov.set_xticks(range(len(phases)))
    ax_cov.set_xticklabels(axis_labels, fontsize=9, fontweight="bold")
    ax_cov.xaxis.set_ticks_position("top")
    ax_cov.set_title("Field Coverage", fontsize=11, fontweight="bold", pad=25)
    for i in range(len(field_labels)):
        for j in range(len(phases)):
            val = cov_matrix[i, j]
            ax_cov.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if val < 0.5 else "black")
    # fig.colorbar(im_v, ax=ax_cov, shrink=0.7, location="bottom", pad=0.06)

    # fig.suptitle(f"{label} — Attribute Confidence & Coverage", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_calibration_scatter(data, phases, fields, label, filename, judge_scores):
    """Scatter: attr_confidence (x) vs judge_accuracy (y), one point per filled field.

    Diagonal y=x marks perfect calibration. Points above = underconfident (filled
    better than confidence suggested); below = overconfident. Informative even
    when N=1, because it maps over the 30+ filled fields of a single card.
    """
    color_cycle = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db", "#9b59b6"]
    phase_colors = {p: color_cycle[i % len(color_cycle)] for i, p in enumerate(phases)}

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, alpha=0.5,
            label="perfect calibration (y=x)")
    for phase in phases:
        xs, ys, labels = [], [], []
        for f in fields:
            conf = _get(data, phase, f, "attr_confidence")
            cov = _get(data, phase, f, "coverage_pct")
            if cov <= 0:
                continue  # skip unfilled fields
            acc = judge_scores.get((phase, f))
            if acc is None:
                continue  # skip unjudged fields
            xs.append(conf)
            ys.append(acc)
            labels.append(f)
        ax.scatter(xs, ys, label=phase, color=phase_colors[phase], s=60, alpha=0.8,
                   edgecolors="white", linewidth=0.5)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab.replace("_", " "), (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=6, alpha=0.7, color=phase_colors[phase])

    ax.set_xlabel("Attribute Confidence (pipeline's self-assessment)", fontsize=10)
    ax.set_ylabel("Judge Accuracy (external evaluation)", fontsize=10)
    ax.set_title(f"Calibration: Confidence vs Judge Accuracy — {label}",
                 fontweight="bold", fontsize=12)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_mc_runtime(judge_scores: dict, filename: str, out_dir: Path | None = None) -> None:
    """Heatmap of M4 runtime fields — one row per experiment-grounded phase, one column per runtime field."""
    runtime_fields = [
        "runtime_suggested_hardware",
        "runtime_expected_f1_range",
        "runtime_expected_latency_ms",
        "runtime_deployment_maturity",
        "runtime_recommended_min_ram_mb",
        "runtime_inference_cost_class",
        "runtime_expected_total_power_w",
        "runtime_typical_deployment_context",
        "runtime_known_failure_modes",
    ]
    # Pull rows whose field is a runtime_* metric (the phase name is not restrictive)
    runtime_phases = sorted({phase for (phase, field) in judge_scores.keys()
                             if field in runtime_fields})
    if not runtime_phases:
        return

    matrix = []
    for phase in runtime_phases:
        row = []
        for rf in runtime_fields:
            v = judge_scores.get((phase, rf), -1)
            row.append(v)
        matrix.append(row)
    matrix = np.array(matrix)

    col_labels = [rf.replace("runtime_", "") for rf in runtime_fields]

    fig, ax = plt.subplots(figsize=(max(8, len(runtime_fields) * 1.0), max(2.5, len(runtime_phases) * 0.7 + 1)))
    # Render -1 as grey via masked array
    masked = np.ma.masked_where(matrix < 0, matrix)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#d0d0d0")
    im = ax.imshow(masked, cmap=cmap, norm=mcolors.Normalize(vmin=0.0, vmax=1.0), aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=30, ha="left")
    ax.set_yticks(range(len(runtime_phases)))
    ax.set_yticklabels(runtime_phases, fontsize=9)
    ax.xaxis.set_ticks_position("top")
    for i in range(len(runtime_phases)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if val < 0:
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="gray")
            else:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white" if val < 0.4 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Accuracy (exact / tolerance / judge)")
    ax.set_title("Runtime Metrics — Model Cards (M4)", fontsize=12, fontweight="bold", pad=50)
    fig.tight_layout()
    target_dir = out_dir if out_dir is not None else CHART_DIR
    fig.savefig(target_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    CHART_DIR.mkdir(exist_ok=True)
    rows = load_csv()
    data = pivot(rows)
    phases = sorted(set(r["phase"] for r in rows))
    judge_scores = load_judge_scores()

    # Step 1 (HF→Patra): M1, M2, M3. Step 2 (Patra→MLHub): just the new M4 phase.
    # Old "M4: Experiment-Grounded" is legacy and stays out of the step-2 charts.
    STEP2_PHASE = "M4: Patra→MLHub"          # matches the CSV phase column
    STEP2_DISPLAY = "Patra → MLHub"           # short label for chart axes
    step1_phases = [p for p in phases if not p.startswith("M4:")]
    mlhub_phases = [p for p in phases if p == STEP2_PHASE]

    has_mc = _has_data(data, step1_phases, MC_FIELDS)
    has_ds = _has_data(data, step1_phases, DS_FIELDS)
    has_mlhub = _has_data(data, mlhub_phases, MLHUB_FIELDS)
    chart_num = 1

    if has_mc:
        # Chart 01 mixes step 1 (MC_FIELDS) + step 2 (MLHUB_FIELDS) so readers see
        # all phases on one scoreboard. Each phase is measured against its own target schema.
        mixed_phases = step1_phases + ([STEP2_PHASE] if has_mlhub else [])
        phase_field_map = {STEP2_PHASE: (MLHUB_FIELDS, MLHUB_REQUIRED)} if has_mlhub else None
        display_names = {STEP2_PHASE: STEP2_DISPLAY}
        chart_metric_overview(data, mixed_phases, MC_FIELDS, MC_REQUIRED,
                              "Model Cards", f"{chart_num:02d}_mc_metrics.png",
                              judge_scores, phase_field_map=phase_field_map,
                              display_names=display_names)
        print(f"  {chart_num:02d}_mc_metrics.png")
        chart_num += 1
        chart_attr_confidence(data, step1_phases, MC_FIELDS, "Model Cards", f"{chart_num:02d}_mc_attr_confidence.png")
        print(f"  {chart_num:02d}_mc_attr_confidence.png")
        chart_num += 1
        chart_coverage(data, step1_phases, MC_FIELDS, "Model Cards", f"{chart_num:02d}_mc_coverage.png")
        print(f"  {chart_num:02d}_mc_coverage.png")
        chart_num += 1
        chart_conf_vs_coverage(data, step1_phases, MC_FIELDS, "Model Cards", f"{chart_num:02d}_mc_conf_vs_cov.png")
        print(f"  {chart_num:02d}_mc_conf_vs_cov.png")
        chart_num += 1

    if has_ds:
        chart_metric_overview(data, step1_phases, DS_FIELDS, DS_REQUIRED, "Datasheets", f"{chart_num:02d}_ds_metrics.png", judge_scores)
        print(f"  {chart_num:02d}_ds_metrics.png")
        chart_num += 1
        chart_attr_confidence(data, step1_phases, DS_FIELDS, "Datasheets", f"{chart_num:02d}_ds_attr_confidence.png")
        print(f"  {chart_num:02d}_ds_attr_confidence.png")
        chart_num += 1
        chart_coverage(data, step1_phases, DS_FIELDS, "Datasheets", f"{chart_num:02d}_ds_coverage.png")
        print(f"  {chart_num:02d}_ds_coverage.png")
        chart_num += 1
        chart_conf_vs_coverage(data, step1_phases, DS_FIELDS, "Datasheets", f"{chart_num:02d}_ds_conf_vs_cov.png")
        print(f"  {chart_num:02d}_ds_conf_vs_cov.png")
        chart_num += 1

    # Legacy M4 runtime chart — rendered into charts/appendix/ (prior Experiment-Grounded work)
    fixture_exists = (Path(__file__).parent / "data" / "inputs" / "mock_experiments.json").exists()
    has_runtime_judge_rows = any(
        field.startswith("runtime_")
        for (phase, field) in judge_scores.keys()
    )
    if fixture_exists and has_runtime_judge_rows:
        APPENDIX_DIR = CHART_DIR / "appendix"
        APPENDIX_DIR.mkdir(exist_ok=True)
        chart_mc_runtime(judge_scores, "09_mc_runtime_metrics.png", out_dir=APPENDIX_DIR)
        print(f"  appendix/09_mc_runtime_metrics.png")

    # Step 2 charts — Patra → MLHub only, strict phase match
    if has_mlhub:
        # Chart 10 (single-row overview heatmap) not rendered — the M4 row already
        # appears in chart 01 alongside M1/M2/M3, making 10 redundant.
        # Chart 11 combines attribute confidence + field coverage in one figure.
        chart_attr_and_coverage(data, mlhub_phases, MLHUB_FIELDS, "Patra → MLHub",
                                "11_mlhub_attr_and_coverage.png",
                                display_names={STEP2_PHASE: STEP2_DISPLAY})
        print(f"  11_mlhub_attr_and_coverage.png")
        chart_calibration_scatter(data, mlhub_phases, MLHUB_FIELDS, "Patra → MLHub",
                                  "13_mlhub_calibration.png", judge_scores)
        print(f"  13_mlhub_calibration.png")

    print(f"\nCharts saved to {CHART_DIR}/")


if __name__ == "__main__":
    main()
