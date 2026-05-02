"""
Cumulative-over-time evaluation metrics across the 450-trial Monte Carlo sweep.

For each trial index N, recomputes (using only trials 1..N):
  - Repair Accuracy   (running mean of trial['repair_accuracy'])
  - Destructive Rate  (running mean of trial['destructive_action'])
  - Abstention Rate   (running mean of trial['abstention'])
  - Resolution Rate   (running mean of 1[trial['action_outcome']=='correct'])

Curves converge to the aggregate values reported in Table I as N -> 450.

Output: experiments/results/cumulative_metrics.pdf
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"

AGENTS = {
    "Rule-Based":    "rule_based_results.json",
    "AIF (No Gate)": "aif_no_gate_results.json",
    "AURORA":        "aurora_results.json",
}

# Per-agent line style so they remain distinguishable in print.
STYLES = {
    "Rule-Based":    {"color": "#888888", "linestyle": "--"},
    "AIF (No Gate)": {"color": "#d62728", "linestyle": ":"},
    "AURORA":        {"color": "#2ca02c", "linestyle": "-"},
}


def load_trials(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)["trials"]


def cumulative(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)


def metric_curves(trials: list[dict]) -> dict[str, np.ndarray]:
    return {
        "Repair Accuracy": cumulative([t["repair_accuracy"] for t in trials]),
        "Destructive Rate": cumulative([1.0 if t["destructive_action"] else 0.0 for t in trials]),
        "Abstention Rate": cumulative([1.0 if t["abstention"] else 0.0 for t in trials]),
        "Resolution Rate": cumulative([1.0 if t["action_outcome"] == "correct" else 0.0 for t in trials]),
    }


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    metric_names = ["Repair Accuracy", "Destructive Rate", "Abstention Rate", "Resolution Rate"]

    agent_curves = {}
    for label, fname in AGENTS.items():
        path = RESULTS_DIR / fname
        trials = load_trials(path)
        agent_curves[label] = (len(trials), metric_curves(trials))

    for ax, name in zip(axes.flat, metric_names):
        for label, (n, curves) in agent_curves.items():
            ax.plot(np.arange(1, n + 1), curves[name], label=label, lw=1.6, **STYLES[label])
        ax.set_title(name)
        ax.set_xlabel("Trial index $N$")
        ax.set_ylabel(name)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out = RESULTS_DIR / "cumulative_metrics.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")

    # Print final converged values so you can sanity-check against Table I.
    print("\nFinal cumulative values (at N = max):")
    for label, (n, curves) in agent_curves.items():
        finals = {k: float(v[-1]) for k, v in curves.items()}
        print(f"  {label:<14}  N={n:>3}  " + "  ".join(f"{k}={v:.3f}" for k, v in finals.items()))


if __name__ == "__main__":
    main()
