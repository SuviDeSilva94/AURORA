"""
Visualize the OAT sensitivity sweep.

Two figures:

  - sensitivity_curves.pdf : 5-panel grid, one panel per parameter, three
    lines per panel (repair / destructive / abstention). Vertical dashed
    line at the published default. Shows the *shape* of each parameter's
    sensitivity surface.

  - sensitivity_tornado.pdf : single tornado plot per metric, ranking
    parameters by their max−min variation across the swept grid. Tells
    a reviewer which parameters are load-bearing and which are robust at
    a glance.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUMMARY_PATH = RESULTS_DIR / "sensitivity_summary.json"

PRETTY = {
    "tau_base":                 r"$\tau_{\mathrm{base}}$",
    "tau_min":                  r"$\tau_{\min}$",
    "tau_lambda":               r"$\lambda$ (Eq. 11)",
    "vfe_threshold":            r"$\mathcal{F}_{\mathrm{th}}$",
    "ranking_margin_threshold": "ranking margin",
}

METRIC_COLORS = {
    "repair_accuracy":  "#2ca02c",
    "destructive_rate": "#d62728",
    "abstention_rate":  "#1f77b4",
}
METRIC_LABELS = {
    "repair_accuracy":  "Repair accuracy",
    "destructive_rate": "Destructive rate",
    "abstention_rate":  "Abstention rate",
}


def plot_curves(summary: dict) -> None:
    sweeps = summary["sweeps"]
    defaults = summary["defaults"]

    n_params = len(sweeps)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else axes

    for idx, (param, rows) in enumerate(sweeps.items()):
        ax = axes[idx]
        xs = [r["value"] for r in rows]
        for metric, color in METRIC_COLORS.items():
            ys = [r[metric] for r in rows]
            ax.plot(xs, ys, marker="o", lw=1.7, color=color,
                    label=METRIC_LABELS[metric])
        ax.axvline(defaults[param], color="black", ls="--", lw=1.0, alpha=0.55,
                   label=f"default = {defaults[param]:g}")
        ax.set_xlabel(PRETTY.get(param, param))
        ax.set_ylabel("Rate / score")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
        ax.set_title(PRETTY.get(param, param), fontsize=11)

    # Hide any leftover axes if grid is bigger than #params.
    for j in range(len(sweeps), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = RESULTS_DIR / "sensitivity_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_tornado(summary: dict) -> None:
    sweeps = summary["sweeps"]
    params = list(sweeps.keys())
    metrics = ["destructive_rate", "abstention_rate", "repair_accuracy"]

    # max−min per metric per parameter
    ranges: dict[str, list[float]] = {m: [] for m in metrics}
    for p in params:
        for m in metrics:
            vals = [r[m] for r in sweeps[p]]
            ranges[m].append((max(vals) - min(vals)) * 100.0)

    # Sort parameters by destructive_rate variation desc (most load-bearing first).
    order = sorted(range(len(params)),
                   key=lambda i: ranges["destructive_rate"][i], reverse=True)
    params_sorted = [params[i] for i in order]
    pretty_sorted = [PRETTY.get(p, p) for p in params_sorted]
    sorted_ranges = {m: [ranges[m][i] for i in order] for m in metrics}

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(params) + 1.5))
    y = np.arange(len(params))
    height = 0.26
    offsets = {-1: -height, 0: 0.0, 1: height}
    for offset_idx, m in zip([-1, 0, 1], metrics):
        ax.barh(y + offsets[offset_idx], sorted_ranges[m], height,
                color=METRIC_COLORS[m], edgecolor="black", alpha=0.85,
                label=METRIC_LABELS[m])

    ax.set_yticks(y)
    ax.set_yticklabels(pretty_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("Metric range across swept values (percentage points)")
    ax.set_title("AURORA OAT sensitivity — ranked by destructive-rate variation")
    ax.grid(alpha=0.25, axis="x")
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    out = RESULTS_DIR / "sensitivity_tornado.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    plot_curves(summary)
    plot_tornado(summary)


if __name__ == "__main__":
    main()
