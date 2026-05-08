"""
Per-seed bar chart visualizing the variance in AURORA's destructive rate
across BN training seeds.

Reads experiments/results/multi_seed_summary.json and writes:

  - destructive_by_seed.pdf : two-panel figure
      (a) AURORA destructive rate per seed, with mean ± SD band and the
          AIF-no-gate baseline as a flat reference line. The bimodal
          pattern (most seeds at 0%, a few at ~10%) jumps out immediately.
      (b) AURORA abstention rate per seed, showing that seeds with
          elevated destructive rates have correspondingly *lower*
          abstention — the gates are letting more borderline trials
          through, not making more conservative calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUMMARY_PATH = RESULTS_DIR / "multi_seed_summary.json"


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    per_seed = summary["per_seed"]
    seeds = [r["seed"] for r in per_seed]
    aurora_destr = [r["agents"]["aurora"]["destructive_rate"] for r in per_seed]
    aurora_abst = [r["agents"]["aurora"]["abstention_rate"] for r in per_seed]
    aif_destr_mean = float(np.mean(
        [r["agents"]["aif_no_gate"]["destructive_rate"] for r in per_seed]
    ))

    aurora_summary = summary["summary"]["aurora"]["destructive_rate"]
    mean_d = aurora_summary["mean"]
    std_d = aurora_summary["std"]

    n_trials = per_seed[0]["agents"]["aurora"]["n"]

    # Color seeds by failure mode: red if destructive > 0, green otherwise.
    bar_colors = ["#d62728" if d > 0 else "#2ca02c" for d in aurora_destr]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(seeds))

    # --- Panel (a): destructive rate ---
    bars = ax1.bar(x, [d * 100 for d in aurora_destr], color=bar_colors,
                   edgecolor="black", alpha=0.85)
    ax1.axhline(mean_d * 100, color="black", ls="--", lw=1.2,
                label=f"AURORA mean = {mean_d*100:.1f}%")
    ax1.fill_between(
        [-0.5, len(seeds) - 0.5],
        [(mean_d - std_d) * 100] * 2,
        [(mean_d + std_d) * 100] * 2,
        color="gray", alpha=0.18, label=f"± 1 SD ({std_d*100:.1f}%)",
    )
    ax1.axhline(aif_destr_mean * 100, color="red", ls=":", lw=1.5,
                label=f"AIF (No Gate) baseline = {aif_destr_mean*100:.1f}%")
    ax1.set_xlim(-0.5, len(seeds) - 0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in seeds])
    ax1.set_xlabel("BN training seed")
    ax1.set_ylabel("Destructive rate (%)")
    ax1.set_title("(a) AURORA destructive rate per seed")
    ax1.set_ylim(0, max(35, max(aurora_destr) * 100 + 5))
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax1.grid(alpha=0.25, axis="y")

    # Annotate the problematic seeds with their counts
    for i, (b, d) in enumerate(zip(bars, aurora_destr)):
        if d > 0:
            ax1.annotate(
                f"{int(d * n_trials)}/{n_trials}",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8, color="black",
            )

    # --- Panel (b): abstention rate ---
    ax2.bar(x, [a * 100 for a in aurora_abst], color="#1f77b4",
            edgecolor="black", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in seeds])
    ax2.set_xlabel("BN training seed")
    ax2.set_ylabel("Abstention rate (%)")
    ax2.set_title("(b) AURORA abstention rate per seed")
    ax2.set_ylim(0, 80)
    ax2.grid(alpha=0.25, axis="y")
    # Annotate the problematic seeds (lower abstention → gates passing more)
    for i, (a, d) in enumerate(zip(aurora_abst, aurora_destr)):
        if d > 0:
            ax2.annotate("↓ gate slip",
                         xy=(i, a * 100),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=8, color="#d62728",
                         fontweight="bold")

    fig.tight_layout()
    out = RESULTS_DIR / "destructive_by_seed.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # Console summary
    print()
    print(f"Across {len(seeds)} seeds × {n_trials} trials:")
    print(f"  AURORA destructive: {mean_d*100:.1f}% ± {std_d*100:.1f}% "
          f"(range [{min(aurora_destr)*100:.1f}%, {max(aurora_destr)*100:.1f}%])")
    bad = [s for s, d in zip(seeds, aurora_destr) if d > 0]
    print(f"  Problematic seeds: {bad} ({len(bad)}/{len(seeds)})")
    print(f"  AIF (No Gate) baseline: {aif_destr_mean*100:.1f}% (flat across seeds)")


if __name__ == "__main__":
    main()
