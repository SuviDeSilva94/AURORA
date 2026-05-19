"""
Stylized GitHub repository landing-page reference for the artifact.

Used in the Reproducibility appendix to show where the open-source code
lives, parallel in role to Peihan's Fig. 11. Renders a faithful but
stylised version of the repository header, README highlights, and key
directories, rather than a live screenshot, so it is reproducible
without browser-state capture.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scienceplots  # noqa: F401


plt.style.use(["science", "ieee", "no-latex"])

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.linewidth": 0,
})

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # ─── Browser chrome (URL bar) ─────────────────────────────────────────
    chrome = mpatches.FancyBboxPatch(
        (0.3, 7.05), 11.4, 0.65,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=1.0, edgecolor="#bbbbbb", facecolor="#f6f8fa", zorder=2,
    )
    ax.add_patch(chrome)
    # Traffic-light circles
    for i, color in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        c = mpatches.Circle((0.65 + i * 0.32, 7.37), 0.13,
                             facecolor=color, edgecolor="none", zorder=3)
        ax.add_patch(c)
    # URL
    url_box = mpatches.FancyBboxPatch(
        (2.0, 7.18), 9.5, 0.42,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=1.0, edgecolor="#cccccc", facecolor="white", zorder=3,
    )
    ax.add_patch(url_box)
    ax.text(2.2, 7.39, "github.com/SuviDeSilva94/AURORA",
            ha="left", va="center", fontsize=9.5, color="#444444",
            family="monospace")

    # ─── Repository header ────────────────────────────────────────────────
    header = mpatches.FancyBboxPatch(
        (0.3, 5.95), 11.4, 0.95,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        linewidth=1.0, edgecolor="#cccccc", facecolor="#ffffff", zorder=2,
    )
    ax.add_patch(header)
    # Repo path
    ax.text(0.6, 6.65, "SuviDeSilva94 / AURORA",
            ha="left", va="center", fontsize=12, fontweight="bold",
            color="#0969da")
    ax.text(0.6, 6.22, "Public",
            ha="left", va="center", fontsize=8.5, color="#57606a",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffff",
                       edgecolor="#d0d7de", linewidth=0.8))
    # Stars / Forks / Watch (stylised)
    for i, (label, value) in enumerate([("Star", "0"),
                                          ("Fork", "0"),
                                          ("Watch", "1")]):
        x = 7.5 + i * 1.4
        box = mpatches.FancyBboxPatch(
            (x, 6.32), 1.2, 0.4,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            linewidth=0.8, edgecolor="#d0d7de", facecolor="#f6f8fa", zorder=3,
        )
        ax.add_patch(box)
        ax.text(x + 0.6, 6.52, f"{label}  {value}",
                ha="center", va="center", fontsize=8.5, color="#24292f")

    # ─── Description line ────────────────────────────────────────────────
    ax.text(0.6, 5.62,
            "AURORA — Dual-gated active inference for trustworthy self-healing on the Computing Continuum.",
            ha="left", va="center", fontsize=9.5, color="#24292f")
    # Topic tags
    topics = ["active-inference", "bayesian-networks", "edge-computing",
               "self-healing", "computing-continuum"]
    tx = 0.6
    for t in topics:
        w = 0.15 + 0.13 * len(t)
        tag = mpatches.FancyBboxPatch(
            (tx, 5.18), w, 0.32,
            boxstyle="round,pad=0.0,rounding_size=0.16",
            linewidth=0.5, edgecolor="#0969da", facecolor="#ddf4ff",
            alpha=0.9, zorder=3,
        )
        ax.add_patch(tag)
        ax.text(tx + w/2, 5.34, t,
                ha="center", va="center", fontsize=8, color="#0969da")
        tx += w + 0.12

    # ─── README highlights (left pane) ───────────────────────────────────
    pane = mpatches.FancyBboxPatch(
        (0.3, 0.5), 7.0, 4.3,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        linewidth=1.0, edgecolor="#d0d7de", facecolor="#ffffff", zorder=2,
    )
    ax.add_patch(pane)
    ax.text(0.55, 4.55, "README.md", ha="left", va="center",
            fontsize=10, fontweight="bold", color="#24292f")
    ax.text(0.55, 4.18, "# AURORA: Uncertainty-aware resilience micro-agent",
            ha="left", va="center", fontsize=9.5, color="#24292f",
            family="monospace")
    readme_lines = [
        "Reference implementation of the dual-gated active-inference",
        "framework evaluated in the companion thesis.",
        "Reproduces all four-workload results:",
        "",
        "  * CCTV streaming (synthetic):    30,006 trials",
        "  * Motor monitoring (synthetic):  30,006 trials",
        "  * AI4I 2020 (real, UCI):           264 trials",
        "  * Steel Plates Faults (real, UCI): 621 trials",
        "",
        "## Quick start",
        "  $ pip install -r requirements.txt",
        "  $ python experiments/run_comparison.py",
        "",
        "## License",
        "  MIT",
    ]
    for i, line in enumerate(readme_lines):
        ax.text(0.55, 3.85 - i * 0.22, line, ha="left", va="center",
                fontsize=8.5, color="#24292f", family="monospace")

    # ─── File tree (right pane) ──────────────────────────────────────────
    tree = mpatches.FancyBboxPatch(
        (7.5, 0.5), 4.2, 4.3,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        linewidth=1.0, edgecolor="#d0d7de", facecolor="#ffffff", zorder=2,
    )
    ax.add_patch(tree)
    ax.text(7.75, 4.55, "Repository layout", ha="left", va="center",
            fontsize=10, fontweight="bold", color="#24292f")
    tree_lines = [
        "AURORA/",
        "  src/aif/",
        "    bayesian_network.py",
        "    two_stage_gate.py",
        "    vfe_compute.py",
        "    root_cause_analyzer.py",
        "  experiments/",
        "    run_comparison.py",
        "    run_motor_comparison.py",
        "    run_real_data_comparison.py",
        "    run_steel_plates_comparison.py",
        "    results/",
        "  tests/",
        "    test_aif.py",
        "  requirements.txt",
        "  LICENSE",
    ]
    for i, line in enumerate(tree_lines):
        ax.text(7.75, 4.22 - i * 0.22, line, ha="left", va="center",
                fontsize=8.5, color="#24292f", family="monospace")

    # ─── Title ────────────────────────────────────────────────────────────
    ax.text(6.0, 7.9, "AURORA artifact repository (GitHub)",
            ha="center", va="bottom", fontsize=12.5, fontweight="bold")

    fig.tight_layout(pad=0.3)
    out = RESULTS / "aurora_github_repo.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
