"""
Plot the saturation curve for repair accuracy, destructive rate, and
abstention rate as a function of trial count N.

Reads ``experiments/results/saturation_summary.json`` and produces:

  - saturation_curve.pdf   : 3-panel figure, one panel per metric, per-agent
                             lines with 95% CI shaded bands. Vertical dashed
                             line at the chosen N=450, plus a second line at
                             the power-analysis n* required to detect the
                             headline destructive-rate gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUMMARY_PATH = RESULTS_DIR / "saturation_summary.json"

AGENT_LABELS = {
    "rule_based":  "Rule-Based",
    "aif_no_gate": "AIF (No Gate)",
    "aurora":      "AURORA",
}
AGENT_COLORS = {
    "rule_based":  "#888888",
    "aif_no_gate": "#d62728",
    "aurora":      "#2ca02c",
}


def _series(rows, key, sub):
    ns = [r["n"] for r in rows]
    if sub == "mean":
        ys = [r[key]["mean"] for r in rows]
    else:
        ys = [r[key]["p_hat"] for r in rows]
    los = [r[key]["ci_low"] for r in rows]
    his = [r[key]["ci_high"] for r in rows]
    return np.array(ns), np.array(ys), np.array(los), np.array(his)


def _panel(ax, summary, metric_key, sub_key, ylabel, ylim=None):
    for agent_key, label in AGENT_LABELS.items():
        rows = summary["agents"].get(agent_key, [])
        if not rows:
            continue
        ns, ys, lo, hi = _series(rows, metric_key, sub_key)
        c = AGENT_COLORS[agent_key]
        ax.plot(ns, ys, marker="o", lw=1.6, color=c, label=label, ms=4)
        ax.fill_between(ns, lo, hi, color=c, alpha=0.18, linewidth=0)
    ax.set_xlabel("Trials per agent (N)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, which="both")
    # Reference lines: N=450 (paper choice) and power-analysis n*
    ax.axvline(450, color="black", ls="--", lw=1.0, alpha=0.6)
    nstar = summary["power_analysis"]["destructive_aif_vs_aurora"]["min_n_per_arm"]
    ax.axvline(nstar, color="red", ls=":", lw=1.0, alpha=0.7)


def main() -> None:
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    _panel(axes[0], summary, "repair_accuracy", "mean",
           "Repair accuracy", ylim=(0.0, 1.0))
    axes[0].set_title("(a) Repair accuracy", fontsize=11)

    _panel(axes[1], summary, "destructive_rate", "p_hat",
           "Destructive action rate", ylim=(-0.02, 0.45))
    axes[1].set_title("(b) Destructive rate", fontsize=11)

    _panel(axes[2], summary, "abstention_rate", "p_hat",
           "Abstention rate", ylim=(-0.02, 1.0))
    axes[2].set_title("(c) Abstention rate", fontsize=11)

    # Single shared legend at the figure level
    handles, labels = axes[0].get_legend_handles_labels()
    nstar = summary["power_analysis"]["destructive_aif_vs_aurora"]["min_n_per_arm"]
    extra_handles = [
        plt.Line2D([0], [0], color="black", ls="--", lw=1.0,
                   label="paper N=450"),
        plt.Line2D([0], [0], color="red", ls=":", lw=1.0,
                   label=f"power-analysis n*={nstar}"),
    ]
    fig.legend(handles + extra_handles, labels + [h.get_label() for h in extra_handles],
               loc="lower center", ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = RESULTS_DIR / "saturation_curve.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
