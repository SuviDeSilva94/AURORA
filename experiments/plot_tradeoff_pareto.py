"""
Safety/resolution Pareto plot for the 30,006-trial Monte Carlo sweep.

Plots each agent as a point in (destructive-rate, resolution-rate) space
with Wilson 95% confidence intervals as error bars. The plot makes the
central thesis trade-off visible at a glance: AURORA Pareto-dominates the
rule-based baseline (lower destructive and comparable/higher resolution)
and is Pareto-incomparable with the un-gated AIF baseline (lower
destructive but lower resolution — the explicit safety-vs-autonomy trade).

Output:
  experiments/results/tradeoff_pareto.pdf   — standalone single-panel figure

Inputs (no new runs needed):
  experiments/results/rule_based_results.json
  experiments/results/aif_no_gate_results.json
  experiments/results/aurora_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers "science" style)

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
    "Rule-Based": "rule_based_results.json",
    "AIF (No Gate)": "aif_no_gate_results.json",
    "AURORA": "aurora_results.json",
}

# Consistent colours with the other thesis figures (plot_distributions.py).
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
    """Return (n_trials, n_destructive, n_correct) for one agent JSON."""
    with open(path) as f:
        data = json.load(f)
    trials = data["trials"]
    n = len(trials)
    n_destructive = sum(1 for t in trials if t.get("destructive_action"))
    n_correct = sum(1 for t in trials if t.get("action_outcome") == "correct")
    return {"n": n, "n_destructive": n_destructive, "n_correct": n_correct}


def compute_points():
    """Compute (dest_rate, res_rate) with Wilson 95% CI for each agent."""
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

    # Pareto-direction shading: the lower-right region is the Pareto-best
    # frontier (low destructive, high resolution). Shade lightly so the
    # reader's eye is drawn toward the desired corner.
    ax.axhspan(0.6, 1.0, xmin=0, xmax=0.05, alpha=0.07, color="#2ca02c", zorder=0)

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

    # Direction-of-improvement arrow in the upper-left quadrant.
    ax.annotate(
        "",
        xy=(2.0, 65.0),
        xytext=(20.0, 50.0),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.6),
    )
    ax.text(
        11.0,
        60.5,
        "Pareto-better",
        rotation=-30,
        color="#2ca02c",
        fontsize=11,
        ha="center",
        va="center",
    )

    # Iso-line at zero destructive: the conceptual "safety axis".
    ax.axvline(0.0, color="#cccccc", linestyle=":", linewidth=1.0, zorder=1)

    ax.set_xlabel("Destructive-action rate (%)")
    ax.set_ylabel("Resolution rate (%)")
    ax.set_xlim(-2, 38)
    ax.set_ylim(20, 80)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    ax.legend(loc="lower right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    out = RESULTS_DIR / "tradeoff_pareto.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")

    # Also emit the numerical points so they can be quoted directly in the
    # thesis caption without re-deriving from the JSONs.
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
    summary_path = RESULTS_DIR / "tradeoff_pareto_points.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
