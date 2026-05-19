"""
Evolution of self-healing approaches in the Computing Continuum
(publication-quality redesign).

Horizontal progression: rule-based threshold controllers → probabilistic
active inference → AURORA dual-gated AIF. Three columns with header,
strength, limitation, and characteristic-references row. Designed for
Extended Background §2.9 State of the Art.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import scienceplots  # noqa: F401


plt.style.use(["science", "ieee", "no-latex"])

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.linewidth": 0,
})

RESULTS = Path(__file__).resolve().parent / "results"


def tier_column(ax, x, *, title, examples, strength, limitation,
                 head_color, body_color, edge_color, lw=1.3):
    """Render one tier as a stacked column with header + 3 body cells."""
    w = 3.2
    cell_h = 0.95
    # Header
    head_y = 4.55
    head = mpatches.FancyBboxPatch(
        (x - w/2, head_y - cell_h/2), w, cell_h,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        linewidth=lw, edgecolor=edge_color, facecolor=head_color, zorder=2,
    )
    ax.add_patch(head)
    ax.text(x, head_y + 0.18, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#222222")
    ax.text(x, head_y - 0.22, examples, ha="center", va="center",
            fontsize=8.5, color="#444444", fontstyle="italic")

    # Strength cell
    str_y = 3.05
    cell = mpatches.FancyBboxPatch(
        (x - w/2, str_y - cell_h/2), w, cell_h,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=1.0, edgecolor="#888888", facecolor=body_color,
        alpha=0.55, zorder=2,
    )
    ax.add_patch(cell)
    ax.text(x, str_y + 0.32, "Strength", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#1a6b30")
    ax.text(x, str_y - 0.18, strength, ha="center", va="center",
            fontsize=8.5, color="#222222")

    # Limitation cell
    lim_y = 1.85
    cell = mpatches.FancyBboxPatch(
        (x - w/2, lim_y - cell_h/2), w, cell_h,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=1.0, edgecolor="#888888", facecolor="#fbe9e7",
        alpha=0.6, zorder=2,
    )
    ax.add_patch(cell)
    ax.text(x, lim_y + 0.32, "Limitation", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#a02020")
    ax.text(x, lim_y - 0.18, limitation, ha="center", va="center",
            fontsize=8.5, color="#222222")


def progression_arrow(ax, x0, x1, y, label):
    arr = FancyArrowPatch(
        (x0, y), (x1, y),
        arrowstyle="-|>", mutation_scale=18,
        color="#444444", lw=1.6, shrinkA=4, shrinkB=4, zorder=3,
    )
    ax.add_patch(arr)
    # Label on top of the arrow
    mx = (x0 + x1) / 2
    ax.text(mx, y + 0.35, label, ha="center", va="bottom",
            fontsize=8.5, color="#222222", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                       edgecolor="#bbbbbb", linewidth=0.6))


def main():
    fig, ax = plt.subplots(figsize=(12.5, 5.3))
    ax.set_xlim(-0.2, 13)
    ax.set_ylim(-0.2, 5.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Tier 1: Rule-based controllers
    tier_column(
        ax, 2.0,
        title="1. Rule-based threshold\ncontrollers",
        examples="Drain-style parsers,\nthreshold orchestrators",
        strength="Fast, interpretable,\nlow edge footprint",
        limitation="No causal model;\nblind to fault ambiguity\nand cascade effects",
        head_color="#fae4c8", body_color="#fff4e3", edge_color="#a06a20",
    )

    # Tier 2: Probabilistic AIF
    tier_column(
        ax, 6.4,
        title="2. Probabilistic\nactive inference",
        examples="Sedlak et al., PAIR-Agent,\nSymphony, NeSy-Edge",
        strength="Causal diagnosis via\nBayesian network +\nMarkov-blanket reasoning",
        limitation="Acts under high\nuncertainty; no formal\nsafety bound on execution",
        head_color="#d8e6f5", body_color="#eef4fb", edge_color="#1f4e89",
    )

    # Tier 3: AURORA dual-gate
    tier_column(
        ax, 10.8,
        title="3. Dual-gated active inference",
        examples="AURORA (this work)",
        strength="Causal substrate +\nbounded certainty +\nVFE epistemic check;\nformal abstention",
        limitation="Resolution traded for\nsafety; absolute 0% destructive\nonly on synthetic data",
        head_color="#d6ecd1", body_color="#ecf6e8", edge_color="#1a6b30",
    )

    # Progression arrows between tiers
    progression_arrow(ax, 3.7, 4.7, 4.55, "add causal model")
    progression_arrow(ax, 8.1, 9.1, 4.55, "bound execution\nby safety primitive")

    # Bottom axis (progression)
    ax.annotate(
        "", xy=(12.6, 0.3), xytext=(0.3, 0.3),
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9),
    )
    ax.text(6.4, 0.05,
            "Increasing safety guarantees on Pi-class edge hardware",
            ha="center", va="bottom", fontsize=9, color="#666666",
            fontstyle="italic")

    # Title
    ax.text(6.4, 5.45,
            "Evolution of self-healing approaches in the Computing Continuum",
            ha="center", va="bottom", fontsize=12.5, fontweight="bold")

    fig.tight_layout(pad=0.3)
    out = RESULTS / "aurora_evolution.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
