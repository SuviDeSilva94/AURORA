"""
Three-gap framing diagram for §1.2 Motivation.

Visualises the three concurrent shortcomings AURORA addresses:
  Gap 1 — diagnosis without causal grounding
  Gap 2 — execution without bounded certainty
  Gap 3 — incompatibility with Pi-class hardware budgets

Each gap is shown as a "problem" cell pointing to its corresponding
AURORA mechanism. Adapted from Peihan's two-axis Data Gap / Trust Gap
framing but specialised to AURORA's actual three-gap structure.
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


def card(ax, x, y, w, h, header, body, *,
          head_color, body_color, edge="#222222", lw=1.2):
    """Two-tone card: coloured header strip + body cell."""
    head_h = 0.55
    head = mpatches.FancyBboxPatch(
        (x - w/2, y + h/2 - head_h), w, head_h,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=lw, edgecolor=edge, facecolor=head_color, zorder=3,
    )
    ax.add_patch(head)
    ax.text(x, y + h/2 - head_h/2, header,
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="#222222")

    body_box = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h - head_h,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=lw, edgecolor=edge, facecolor=body_color, zorder=2,
    )
    ax.add_patch(body_box)
    ax.text(x, y - 0.18, body, ha="center", va="center",
            fontsize=8.5, color="#222222")


def arrow_horizontal(ax, p0, p1, *, label, color="#444444"):
    arr = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=14,
        color=color, lw=1.4, shrinkA=8, shrinkB=8, zorder=4,
    )
    ax.add_patch(arr)
    mx = (p0[0] + p1[0]) / 2
    my = (p0[1] + p1[1]) / 2 + 0.22
    ax.text(mx, my, label, ha="center", va="center",
            fontsize=8.5, color="#222222", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                       edgecolor="#bbbbbb", linewidth=0.6))


def main():
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    ax.set_xlim(-0.3, 13)
    ax.set_ylim(0, 6.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Column 1: Problem (red)
    # Column 2: AURORA mechanism (green)
    # Three rows = three gaps

    prob_x = 2.4
    mech_x = 10.0
    card_w = 4.0
    card_h = 1.35

    row_y = [4.65, 3.0, 1.35]
    gap_titles = ["Gap 1", "Gap 2", "Gap 3"]
    gap_problems = [
        ("Diagnosis without causal grounding",
         "Threshold lookups treat symptoms\nas causes; correlation-based monitoring\ncannot disambiguate fault classes"),
        ("Execution without bounded certainty",
         "Ungated active-inference agents dispatch\nactions while holding high epistemic\nuncertainty; destructive autonomy"),
        ("Incompatibility with Pi-class budgets",
         "Monolithic LLM controllers and cloud\norchestrators exceed the memory, latency,\nand power budgets of edge hardware"),
    ]
    aurora_mechanisms = [
        ("Markov-blanket-bounded BN",
         "Variable elimination on the symptom's\nMarkov blanket; do-calculus interventional\ncontrast $\\Delta(C)$ ranks candidate causes"),
        ("Dual-gated safety mechanism",
         "Gate 1 ($P_{\\max} \\geq \\tau_{\\text{eff}}$) + Gate 2\n($\\mathcal{F}_{\\text{total}} < \\mathcal{F}_{\\text{th}}$) enforce two orthogonal\nbounds before any action dispatches"),
        ("Parallelised micro-agent runtime",
         "Single-responsibility specialists run\nconcurrently; per-decision MTTR around 3.4 ms\non a laptop reference, well inside the\nPi-4 budget"),
    ]

    # Render problem-mechanism row by row
    for i, (gap, (p_title, p_body), (m_title, m_body)) in enumerate(
        zip(gap_titles, gap_problems, aurora_mechanisms)
    ):
        y = row_y[i]
        # Gap number badge
        circ = mpatches.Circle((0.55, y), 0.32,
                                facecolor="#444444", edgecolor="#222222",
                                linewidth=1.0, zorder=3)
        ax.add_patch(circ)
        ax.text(0.55, y, gap, ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")

        # Problem card
        card(ax, prob_x, y, card_w, card_h,
             p_title, p_body,
             head_color="#fbe9e7", body_color="#fdf3f1", edge="#a02020")

        # Arrow with mechanism label
        arrow_horizontal(ax, (prob_x + card_w/2, y),
                          (mech_x - card_w/2, y),
                          label="AURORA addresses by")

        # Mechanism card
        card(ax, mech_x, y, card_w, card_h,
             m_title, m_body,
             head_color="#d8ecd4", body_color="#eef6ec", edge="#1a6b30")

    # Column headers
    ax.text(prob_x, 5.65, "Shortcoming",
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color="#a02020")
    ax.text(mech_x, 5.65, "AURORA mechanism",
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color="#1a6b30")

    # Title
    ax.text(6.2, 5.95,
            "Three concurrent shortcomings that AURORA addresses",
            ha="center", va="bottom", fontsize=12.5, fontweight="bold")

    # Footer
    ax.text(6.2, 0.15,
            "Section 1.2 elaborates each gap; the corresponding mechanisms are developed in Chapter 4",
            ha="center", va="bottom", fontsize=8.5, color="#666666",
            fontstyle="italic")

    fig.tight_layout(pad=0.3)
    out = RESULTS / "aurora_three_gap.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
