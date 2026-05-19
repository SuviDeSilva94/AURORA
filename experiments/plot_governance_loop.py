"""
AURORA Conceptual Governance Loop diagram (publication-quality redesign).

High-level boxes-and-arrows view of the AURORA framework: telemetry
observation, BN-bounded diagnosis, dual-gate safety check, and the two
downstream paths (local execution or fog-tier abstention). Sits at the
head of Chapter 4 Architecture, before the detailed five-phase pipeline.
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
    "font.size": 10,
    "axes.linewidth": 0,
})


RESULTS = Path(__file__).resolve().parent / "results"


# ─── Helpers ────────────────────────────────────────────────────────────────

def rounded_box(ax, x, y, w, h, *, facecolor, edgecolor, lw=1.0, alpha=1.0):
    box = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        linewidth=lw, edgecolor=edgecolor, facecolor=facecolor,
        alpha=alpha, zorder=2,
    )
    ax.add_patch(box)


def stage_band(ax, x_center, y_top, y_bot, w, label, fill):
    """A faint background band grouping multiple boxes under a stage label."""
    box = mpatches.FancyBboxPatch(
        (x_center - w/2, y_bot), w, y_top - y_bot,
        boxstyle="round,pad=0.0,rounding_size=0.12",
        linewidth=0, facecolor=fill, alpha=0.25, zorder=1,
    )
    ax.add_patch(box)
    # Stage label, vertical on the left of the band
    ax.text(x_center - w/2 - 0.18, (y_top + y_bot)/2, label,
            ha="right", va="center", fontsize=9, color="#444444",
            fontweight="bold", rotation=90)


def box(ax, x, y, w, h, title, body=None, color="#e8e8e8", edge="#333333",
        lw=1.0, title_size=10, body_size=8.5):
    rounded_box(ax, x, y, w, h, facecolor=color, edgecolor=edge, lw=lw)
    if body:
        # title on top, body below
        ax.text(x, y + h/4, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", zorder=3)
        ax.text(x, y - h/4, body, ha="center", va="center",
                fontsize=body_size, color="#222222", zorder=3)
    else:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=title_size, zorder=3)


def arrow(ax, p0, p1, *, color="#222222", lw=1.4, style="-|>", label=None,
          label_offset=(0, 0.0), label_color=None):
    arr = FancyArrowPatch(
        p0, p1,
        arrowstyle=style, mutation_scale=14,
        color=color, lw=lw,
        shrinkA=8, shrinkB=8,
        zorder=4,
    )
    ax.add_patch(arr)
    if label:
        mx = (p0[0] + p1[0]) / 2 + label_offset[0]
        my = (p0[1] + p1[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold",
                color=label_color or color,
                bbox=dict(boxstyle="round,pad=0.18",
                           facecolor="white", edgecolor="none", alpha=0.9))


def curved_arrow(ax, p0, p1, *, rad=0.3, color="#666666", lw=1.0, style="-|>"):
    arr = FancyArrowPatch(
        p0, p1,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style, mutation_scale=12,
        color=color, lw=lw,
        shrinkA=8, shrinkB=8,
        zorder=4,
    )
    ax.add_patch(arr)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.set_xlim(-0.6, 13)
    ax.set_ylim(0, 9.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colour palette (muted, publication-friendly)
    c_observe = "#dee9f5"   # cool blue (observation)
    c_diag    = "#d8ecd4"   # cool green (diagnosis)
    c_gate    = "#fce8b8"   # warm amber (gates)
    c_exec    = "#cce6d8"   # exec green
    c_abst    = "#f4c8c4"   # abstain red
    edge_dark = "#222222"
    edge_exec = "#1a7a3d"
    edge_abst = "#a02020"

    band_w = 12.2
    band_x = 6.0

    # ─── Stage bands (background) ──────────────────────────────────────────
    stage_band(ax, band_x, 8.6, 7.0, band_w, "Observation", "#9ec1de")
    stage_band(ax, band_x, 6.4, 4.0, band_w, "Diagnosis",  "#9bcc91")
    stage_band(ax, band_x, 3.4, 2.0, band_w, "Dual-gate",  "#f5cf6c")
    stage_band(ax, band_x, 1.4, 0.2, band_w, "Action",     "#aacca8")

    # ─── Observation row ──────────────────────────────────────────────────
    box(ax, 2.0, 7.8, 3.2, 0.95,
        "Edge telemetry",
        r"$O_t = \langle$delay, cpu, memory," + "\n" + r"throughput, fps$\rangle$",
        color=c_observe, edge=edge_dark, title_size=10.5, body_size=8.5)
    box(ax, 6.5, 7.8, 3.2, 0.95,
        "Parallel micro-agents",
        "single-responsibility\nanomaly checks",
        color=c_observe, edge=edge_dark)
    box(ax, 11.0, 7.8, 3.2, 0.95,
        "Anomaly vector $A_t$",
        "SLO predicates fired",
        color=c_observe, edge=edge_dark)

    arrow(ax, (3.6, 7.8), (4.9, 7.8))
    arrow(ax, (8.1, 7.8), (9.4, 7.8))

    # ─── Diagnosis row ─────────────────────────────────────────────────────
    box(ax, 3.8, 5.2, 4.0, 1.05,
        "Phase 2 — Root-cause analysis",
        r"Markov-blanket-bounded BN posterior" + "\n" + r"$P(C \mid O_t)$",
        color=c_diag, edge=edge_dark)
    box(ax, 9.6, 5.2, 4.0, 1.05,
        "Composite ranking $\\rho(C)$",
        r"impact $\Delta(C)$ $\;\cdot\;$ evidence $\;\cdot\;$" + "\n"
        + r"path factor $\;\cdot\;$ structural prior",
        color=c_diag, edge=edge_dark)

    # Connector from anomaly vector down to RCA + ranking
    arrow(ax, (11.0, 7.27), (3.8, 5.78))
    arrow(ax, (11.0, 7.27), (9.6, 5.78))
    # Internal flow between RCA and ranking
    arrow(ax, (5.85, 5.2), (7.55, 5.2))

    # ─── Dual gate row ────────────────────────────────────────────────────
    box(ax, 3.8, 2.7, 4.0, 1.0,
        "Gate 1 — Posterior certainty",
        r"$P_{\max} \;\geq\; \tau_{\text{eff}}$",
        color=c_gate, edge="#9b6a00")
    box(ax, 9.6, 2.7, 4.0, 1.0,
        "Gate 2 — VFE bound",
        r"$\mathcal{F}_{\text{total}} \;<\; \mathcal{F}_{\text{th}}$",
        color=c_gate, edge="#9b6a00")

    arrow(ax, (3.8, 4.65), (3.8, 3.25))
    arrow(ax, (9.6, 4.65), (9.6, 3.25))
    arrow(ax, (5.85, 2.7), (7.55, 2.7))

    # ─── Action row ───────────────────────────────────────────────────────
    box(ax, 3.2, 0.85, 4.4, 0.9,
        "Local mitigation",
        r"dispatch $\mathrm{do}(a)$ on the edge node",
        color=c_exec, edge=edge_exec, lw=1.4)
    box(ax, 10.0, 0.85, 4.4, 0.9,
        "Abstain $\\to$ fog tier",
        "serialised diagnostic payload",
        color=c_abst, edge=edge_abst, lw=1.4)

    arrow(ax, (3.8, 2.18), (3.2, 1.32), color=edge_exec, lw=1.6,
          label="both pass", label_offset=(-0.55, 0.05))
    arrow(ax, (9.6, 2.18), (10.0, 1.32), color=edge_abst, lw=1.6,
          label="either fires", label_offset=(0.55, 0.05))

    # ─── Feedback loop (right-hand curl) ──────────────────────────────────
    curved_arrow(ax, (12.15, 0.85), (12.5, 7.8), rad=-0.45,
                  color="#666666", lw=1.0)
    ax.text(12.65, 4.3, "Feedback\nfog re-inference\nBN re-fit",
            ha="left", va="center", fontsize=8.5, color="#444444",
            rotation=90, fontstyle="italic")

    # ─── Title and footer ─────────────────────────────────────────────────
    ax.text(band_x, 9.05, "AURORA Conceptual Governance Loop",
            ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.text(band_x, -0.3,
            "Edge node $\\rightarrow$ Continuum Bridge $\\rightarrow$ fog tier",
            ha="center", va="bottom", fontsize=9, color="#666666",
            fontstyle="italic")

    fig.tight_layout(pad=0.4)
    out = RESULTS / "aurora_governance_loop.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
