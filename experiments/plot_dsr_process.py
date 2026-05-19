"""
DSR Process applied to AURORA (publication-quality flowchart).

Visualises the six-stage Design Science Research process (Peffers et al.,
2007) instantiated for the AURORA artifact. Sits in Method §3.2, parallel
in role to Peihan's Fig. 4 but with AURORA-specific labels.
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


def stage(ax, x, y, w, h, num, title, body, *, head_color, body_color,
          edge="#222222", lw=1.2):
    """Render a numbered DSR stage as a header + body card."""
    # Header strip
    head_h = 0.42
    head = mpatches.FancyBboxPatch(
        (x - w/2, y + h/2 - head_h), w, head_h,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        linewidth=lw, edgecolor=edge, facecolor=head_color, zorder=3,
    )
    ax.add_patch(head)
    ax.text(x - w/2 + 0.2, y + h/2 - head_h/2,
            f"{num}", ha="left", va="center",
            fontsize=12, fontweight="bold", color="#222222")
    ax.text(x + 0.05, y + h/2 - head_h/2, title,
            ha="left", va="center",
            fontsize=10, fontweight="bold", color="#222222")

    # Body
    body_box = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h - head_h,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        linewidth=lw, edgecolor=edge, facecolor=body_color, zorder=2,
    )
    ax.add_patch(body_box)
    ax.text(x, y - 0.18, body, ha="center", va="center",
            fontsize=8.5, color="#222222")


def arrow(ax, p0, p1, *, color="#444444", lw=1.4, style="-|>"):
    arr = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=14,
        color=color, lw=lw, shrinkA=8, shrinkB=8, zorder=4,
    )
    ax.add_patch(arr)


def main():
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.set_xlim(-0.3, 13)
    ax.set_ylim(0, 6.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Six DSR stages, two rows of three
    box_w, box_h = 3.5, 1.55

    # Top row: stages 1, 2, 3
    stage(ax, 2.0, 4.4, box_w, box_h, "1",
          "Problem Identification",
          "Grey failures cause\ndestructive autonomy under\nuncertainty on Pi-class edge",
          head_color="#dee9f5", body_color="#f4f8fc")
    stage(ax, 6.5, 4.4, box_w, box_h, "2",
          "Define Objectives",
          "0% destructive on\nambiguous CCTV faults,\ndecision latency below 10 ms",
          head_color="#dee9f5", body_color="#f4f8fc")
    stage(ax, 11.0, 4.4, box_w, box_h, "3",
          "Design & Develop",
          "Dual-gated AIF with\nMarkov-blanket BN +\nVFE-based gate primitive",
          head_color="#d8ecd4", body_color="#eef6ec")

    # Bottom row: stages 4, 5, 6 (reversed direction so arrow wraps)
    stage(ax, 11.0, 1.6, box_w, box_h, "4",
          "Demonstration",
          "Reference implementation,\n4 workloads, open-source\nartefact + reproducibility",
          head_color="#d8ecd4", body_color="#eef6ec")
    stage(ax, 6.5, 1.6, box_w, box_h, "5",
          "Evaluation",
          "30,006 CCTV + 30,006 motor\n+ 264 AI4I + 621 Steel Plates;\nWilson 95% CIs, statistical tests",
          head_color="#fce8b8", body_color="#fff7e1")
    stage(ax, 2.0, 1.6, box_w, box_h, "6",
          "Communication",
          "This thesis +\nopen-source artifact +\nreproducibility package",
          head_color="#fce8b8", body_color="#fff7e1")

    # Replace LaTeX-style escapes in body strings (we use no-latex mode)
    # Actually we want \\\\ to render as \n, etc. Easier: redraw with raw strings.
    # (Stage 6 body has \\\\ which doesn't render — fix by re-applying clean text)
    for t in ax.texts:
        s = t.get_text()
        if "\\\\" in s or "\\&" in s or "\\%" in s or "$" in s:
            new_s = (s.replace("\\\\", "\n")
                      .replace("\\&", "&")
                      .replace("\\%", "%"))
            t.set_text(new_s)

    # Horizontal arrows in top row (1 → 2 → 3)
    arrow(ax, (3.75, 4.4), (4.75, 4.4))
    arrow(ax, (8.25, 4.4), (9.25, 4.4))

    # Curved arrow: stage 3 → stage 4 (top right → bottom right)
    arrow(ax, (11.0, 3.62), (11.0, 2.38))

    # Horizontal arrows in bottom row (4 ← 5 ← 6 → reading flow)
    # Visually we want: 4 → 5 → 6 with arrows showing rightward flow becomes leftward
    arrow(ax, (9.25, 1.6), (8.25, 1.6))
    arrow(ax, (4.75, 1.6), (3.75, 1.6))

    # Feedback loop: stage 6 → stage 1 (refines problem understanding)
    feedback = FancyArrowPatch(
        (2.0, 2.38), (2.0, 3.62),
        arrowstyle="-|>", mutation_scale=14, color="#888888",
        lw=1.0, linestyle="--", shrinkA=8, shrinkB=8, zorder=4,
    )
    ax.add_patch(feedback)
    ax.text(0.7, 3.0, "refinement\nfeedback", ha="center", va="center",
            fontsize=8.5, color="#666666", fontstyle="italic")

    # Title
    ax.text(6.5, 5.85,
            "Design Science Research process applied to AURORA",
            ha="center", va="bottom", fontsize=12.5, fontweight="bold")

    # Footer
    ax.text(6.5, 0.15,
            "Stages 1–6 follow Peffers et al. (2007); the dashed arrow marks the refinement loop between Communication and re-scoped Problem Identification",
            ha="center", va="bottom", fontsize=8, color="#666666",
            fontstyle="italic")

    fig.tight_layout(pad=0.3)
    out = RESULTS / "aurora_dsr_process.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
