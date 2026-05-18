"""
Motor-monitoring safety/resolution Pareto plot.

Mirrors `plot_tradeoff_pareto.py` for the second workload, allowing
side-by-side reading of the dual-gate's behaviour across two structurally
distinct fault distributions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from src.evaluation.proportion_ci import wilson_interval


plt.style.use(["science", "ieee", "no-latex"])

mpl.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "axes.linewidth": 1.1,
        "lines.linewidth": 1.6,
    }
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"

AGENTS = {
    "Rule-Based": "motor_rule_based_results.json",
    "AIF (No Gate)": "motor_aif_no_gate_results.json",
    "AURORA": "motor_aurora_results.json",
}

COLORS = {
    "Rule-Based": "#888888",
    "AIF (No Gate)": "#d62728",
    "AURORA": "#2ca02c",
}

MARKERS = {
    "Rule-Based": "s",
    "AIF (No Gate)": "^",
    "AURORA": "o",
}


def load_counts(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    trials = data["trials"]
    n = len(trials)
    n_destructive = sum(1 for t in trials if t.get("destructive_action"))
    n_correct = sum(1 for t in trials if t.get("action_outcome") == "correct")
    return {"n": n, "n_destructive": n_destructive, "n_correct": n_correct}


def compute_points():
    points = {}
    for label, fname in AGENTS.items():
        c = load_counts(RESULTS_DIR / fname)
        n = c["n"]
        dest_p = c["n_destructive"] / n
        res_p = c["n_correct"] / n
        dest_lo, dest_hi = wilson_interval(c["n_destructive"], n)
        res_lo, res_hi = wilson_interval(c["n_correct"], n)
        points[label] = {
            "n": n,
            "dest_p": dest_p,
            "dest_err": [[dest_p - dest_lo], [dest_hi - dest_p]],
            "res_p": res_p,
            "res_err": [[res_p - res_lo], [res_hi - res_p]],
        }
    return points


def main():
    points = compute_points()

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for label, p in points.items():
        xerr = [[p["dest_err"][0][0] * 100], [p["dest_err"][1][0] * 100]]
        yerr = [[p["res_err"][0][0] * 100], [p["res_err"][1][0] * 100]]
        ax.errorbar(
            p["dest_p"] * 100,
            p["res_p"] * 100,
            xerr=xerr,
            yerr=yerr,
            fmt=MARKERS[label],
            color=COLORS[label],
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.8,
            elinewidth=1.4,
            capsize=4,
            label=label,
            zorder=3,
        )

    ax.annotate(
        "",
        xy=(3.0, 70.0),
        xytext=(20.0, 45.0),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.6),
    )
    ax.text(
        12.5,
        60.0,
        "Pareto-better",
        rotation=-30,
        color="#2ca02c",
        fontsize=11,
        ha="center",
        va="center",
    )

    ax.axvline(0.0, color="#cccccc", linestyle=":", linewidth=1.0, zorder=1)

    ax.set_xlabel("Destructive-action rate (%)")
    ax.set_ylabel("Resolution rate (%)")
    ax.set_xlim(-2, 38)
    ax.set_ylim(25, 75)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_title("Motor monitoring workload", fontsize=14, pad=8)

    ax.legend(loc="lower right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    out = RESULTS_DIR / "motor_tradeoff_pareto.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")

    summary = {
        label: {
            "n_trials": p["n"],
            "destructive_rate_pct": p["dest_p"] * 100,
            "destructive_ci_pct": [
                (p["dest_p"] - p["dest_err"][0][0]) * 100,
                (p["dest_p"] + p["dest_err"][1][0]) * 100,
            ],
            "resolution_rate_pct": p["res_p"] * 100,
            "resolution_ci_pct": [
                (p["res_p"] - p["res_err"][0][0]) * 100,
                (p["res_p"] + p["res_err"][1][0]) * 100,
            ],
        }
        for label, p in points.items()
    }
    summary_path = RESULTS_DIR / "motor_tradeoff_pareto_points.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
