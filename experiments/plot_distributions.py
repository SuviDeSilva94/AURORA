"""
Per-trial distribution plots (box + violin) for the 450-trial Monte Carlo sweep.

Outputs into experiments/results/:
  - distributions.pdf       : 2x2 master figure with all four panels
  - dist_vfe_by_fault.pdf   : standalone panel (a) - VFE per fault class
  - dist_mttr.pdf           : standalone panel (b) - MTTR per agent (LOG y-axis)
  - dist_certainty.pdf      : standalone panel (c) - posterior certainty
  - dist_score.pdf          : standalone panel (d) - per-trial outcome score

Box plots use Tukey whiskers (1.5*IQR); points beyond are drawn as outliers.
Violin plots show the full kernel-density-estimated distribution.

Panel (b) uses a log y-axis because Rule-Based MTTR (~1 ms) and AIF/AURORA
MTTR (~13 ms) differ by ~10x; a linear axis collapses Rule-Based into a flat
strip. Other panels keep linear axes (their meaningful structure is at a
fixed threshold or in a bounded [0,1] range).

All output is scalable, text-selectable vector PDF.
"""

import json
from pathlib import Path

import scienceplots
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Apply the requested IEEE style with SciencePlots
plt.style.use(["science", "ieee", "no-latex"])

# Custom colors based on the reviewer's request
colors = ['#0C5DA5', '#00B945', '#FF9500', '#845B97', '#474747', '#9E9E9E']


# Aggressive font sizing for IEEE two-column print: caption ~8pt at print scale,
# so axis text needs to render LARGER than the caption when scaled to the
# column. These values are chosen so the included PDF, scaled to a single
# IEEE column (~3.5"), still has tick labels >= 7pt and axis labels >= 8pt.
mpl.rcParams.update({
    "font.size":        16,
    "axes.labelsize":   18,
    "axes.titlesize":   18,
    "xtick.labelsize":  16,
    "ytick.labelsize":  18,
    "legend.fontsize":  15,
    "figure.titlesize": 20,
    "axes.linewidth":   1.1,
    "lines.linewidth":  1.8,
    "axes.prop_cycle": plt.cycler(color=colors),
})

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Ensure consistent width/height for all standalone panels.
FIGSIZE_PANEL = (5.0, 3.5)

AGENTS = {
    "Rule-Based":    "rule_based_results.json",
    "AIF (No Gate)": "aif_no_gate_results.json",
    "AURORA":        "aurora_results.json",
}
COLORS = {
    "Rule-Based":    "#888888",
    "AIF (No Gate)": "#d62728",
    "AURORA":        "#2ca02c",
}
FAULT_COLORS = {
    "network_drop": "#1f77b4",
    "cpu_spike":    "#d62728",
    "memory_leak":  "#9467bd",
}


def load_trials(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)["trials"]


def panel_violin(ax, data_per_agent, ylabel, title=None, ylim=None):
    """Generic violin-with-box-inside, one violin per agent.

    The `title` argument is accepted but intentionally ignored — panel
    headings are rendered by the LaTeX caption, not embedded in the PDF
    (per supervisor review).
    """
    labels = list(data_per_agent.keys())
    data = [np.asarray(data_per_agent[k], dtype=float) for k in labels]

    safe_data = []
    for arr in data:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            arr = np.array([0.0])
        safe_data.append(arr)

    parts = ax.violinplot(safe_data, showmeans=False,
                          showmedians=False, showextrema=False)
    for body, label in zip(parts["bodies"], labels):
        body.set_facecolor(COLORS[label])
        body.set_alpha(0.45)
        body.set_edgecolor("black")
    ax.boxplot(
        safe_data,
        widths=0.18,
        patch_artist=False,
        medianprops={"color": "black", "linewidth": 1.4},
        flierprops={"marker": "x", "markersize": 3, "alpha": 0.6},
    )
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(False)


def panel_vfe_by_fault(ax, aurora_trials, title=None):
    """AURORA only: VFE distribution split by injected fault class.

    The `title` argument is accepted but intentionally ignored — panel
    headings are rendered by the LaTeX caption, not embedded in the PDF.
    """
    groups = {"network_drop": [], "cpu_spike": [], "memory_leak": []}
    for t in aurora_trials:
        f = t.get("fault_type")
        v = t.get("vfe_value")
        if f in groups and v is not None:
            groups[f].append(v)
    labels = ["network_drop", "cpu_spike", "memory_leak"]
    data = [np.asarray(groups[k], dtype=float) for k in labels]
    parts = ax.violinplot(data, showmeans=False,
                          showmedians=False, showextrema=False)
    for body, label in zip(parts["bodies"], labels):
        body.set_facecolor(FAULT_COLORS[label])
        body.set_alpha(0.5)
        body.set_edgecolor("black")
    ax.boxplot(
        data,
        widths=0.18,
        patch_artist=False,
        medianprops={"color": "black", "linewidth": 1.4},
        flierprops={"marker": "x", "markersize": 3, "alpha": 0.6},
    )
    ax.axhline(3.85, color="red", ls="--", lw=1.2)
    ax.text(0.55, 3.95, r"$\mathcal{F}_{\rm th}=3.85$", color="red")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Network Drop", "CPU Spike", "Memory Leak"])
    ax.set_ylabel(r"VFE score $\mathcal{F}$")
    ax.grid(False)
    ax.legend(
        loc="upper left",
        framealpha=0.85, facecolor="white", edgecolor="0.7", fancybox=True,
    )


def apply_linear_mttr(ax) -> None:
    """Linear y-axis for MTTR. Rule-Based clusters near 1 ms, AIF/AURORA near
    13 ms; the linear axis honestly shows the ~10x gap rather than hiding it
    under a log axis. Consistent with all other panels (linear scale)."""
    ax.set_ylim(-0.002, 0.05)
    ax.grid(False)


def fmt_quartiles(a: np.ndarray) -> str:
    if a.size == 0:
        return "n/a"
    return (
        f"median={np.median(a):.4f}, "
        f"IQR=[{np.quantile(a, 0.25):.4f}, {np.quantile(a, 0.75):.4f}], "
        f"min={a.min():.4f}, max={a.max():.4f}"
    )


def remove_border_yticks(ax) -> None:
    """Prune problematic y-ticks for export-quality PDFs.

    Goals:
      1) Never show a y tick *on* the plot border (ticks at exactly ylim).
      2) Never show negative y ticks.
      3) Avoid minor y ticks that can still sit on the border.

    We implement this by generating candidate ticks via a MaxNLocator with
    end pruning, then filtering.
    """
    if ax.get_yscale() != "linear":
        return

    ax.yaxis.set_minor_locator(mticker.NullLocator())

    ymin, ymax = ax.get_ylim()
    span = max(abs(ymax - ymin), 1e-12)
    atol = 1e-6 * span

    locator = mticker.MaxNLocator(nbins=6, prune="both")
    ticks = np.asarray(locator.tick_values(ymin, ymax), dtype=float)
    ticks = ticks[np.isfinite(ticks)]

    # No negative ticks (also normalizes -0.0 to 0.0).
    ticks[np.isclose(ticks, 0.0, rtol=0.0, atol=atol)] = 0.0
    ticks = ticks[ticks >= 0.0 - atol]

    # No ticks exactly at the borders.
    ticks = ticks[~np.isclose(ticks, ymin, rtol=0.0, atol=atol)]
    ticks = ticks[~np.isclose(ticks, ymax, rtol=0.0, atol=atol)]

    # Only override if we still have a sensible set.
    if ticks.size >= 2:
        ax.set_yticks(ticks)


def main() -> None:
    trials = {label: load_trials(RESULTS_DIR / fname)
              for label, fname in AGENTS.items()}

    mttr = {k: [t["mttr_seconds"] for t in v] for k, v in trials.items()}
    cert = {
        k: [t["certainty_score"]
            for t in v if t.get("certainty_score") is not None]
        for k, v in trials.items()
    }
    score = {k: [t["repair_accuracy"] for t in v] for k, v in trials.items()}

    # ---------- Combined 2x2 master ----------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    panel_vfe_by_fault(axes[0, 0], trials["AURORA"])
    axes[0, 0].margins(y=0.03)
    remove_border_yticks(axes[0, 0])

    panel_violin(axes[0, 1], mttr,
                 ylabel="MTTR (s)",
                 title="(b) Decision latency per agent")
    apply_linear_mttr(axes[0, 1])
    remove_border_yticks(axes[0, 1])

    panel_violin(axes[1, 0], cert,
                 ylabel=r"Posterior certainty $P_{\max}$",
                 title="(c) Posterior certainty per agent",
                 ylim=(-0.02, 1.05))
    axes[1, 0].axhline(0.70, color="blue", ls="--", lw=1.2, alpha=0.7)
    axes[1, 0].text(0.55, 0.72, r"$\tau=0.70$", color="blue")
    remove_border_yticks(axes[1, 0])

    panel_violin(axes[1, 1], score,
                 ylabel=r"Accuracy $S_i$",
                 title="(d) Accuracy per agent",
                 ylim=(-0.05, 1.05))
    remove_border_yticks(axes[1, 1])

    fig.tight_layout()
    out = RESULTS_DIR / "distributions.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # ---------- Per-panel standalone PDFs ----------
    # (a) VFE by fault class
    f_a, ax_a = plt.subplots(figsize=FIGSIZE_PANEL)
    panel_vfe_by_fault(
        ax_a, trials["AURORA"], title="VFE distribution by fault class (AURORA)")
    ax_a.margins(y=0.03)
    remove_border_yticks(ax_a)
    f_a.tight_layout()
    out_a = RESULTS_DIR / "dist_vfe_by_fault.pdf"
    f_a.savefig(out_a, bbox_inches="tight")
    plt.close(f_a)
    print(f"Wrote {out_a}")

    # (b) MTTR per agent (LOG y)
    f_b, ax_b = plt.subplots(figsize=FIGSIZE_PANEL)
    panel_violin(ax_b, mttr,
                 ylabel="MTTR (s)",
                 title="Decision latency per agent")
    apply_linear_mttr(ax_b)
    remove_border_yticks(ax_b)
    f_b.tight_layout()
    out_b = RESULTS_DIR / "dist_mttr.pdf"
    f_b.savefig(out_b, bbox_inches="tight")
    plt.close(f_b)
    print(f"Wrote {out_b}")

    # (c) Posterior certainty per agent
    f_c, ax_c = plt.subplots(figsize=FIGSIZE_PANEL)
    panel_violin(ax_c, cert,
                 ylabel=r"Posterior certainty $P_{\max}$",
                 title="Posterior certainty per agent",
                 ylim=(-0.02, 1.05))
    ax_c.axhline(0.70, color="blue", ls="--", lw=1.2, alpha=0.7)
    ax_c.text(0.55, 0.72, r"$\tau=0.70$", color="blue")
    remove_border_yticks(ax_c)
    f_c.tight_layout()
    out_c = RESULTS_DIR / "dist_certainty.pdf"
    f_c.savefig(out_c, bbox_inches="tight")
    plt.close(f_c)
    print(f"Wrote {out_c}")

    # (d) Per-trial accuracy score per agent
    f_d, ax_d = plt.subplots(figsize=FIGSIZE_PANEL)
    panel_violin(ax_d, score,
                 ylabel=r"Accuracy $S_i$",
                 title="Accuracy per agent",
                 ylim=(-0.05, 1.05))
    remove_border_yticks(ax_d)
    f_d.tight_layout()
    out_d = RESULTS_DIR / "dist_score.pdf"
    f_d.savefig(out_d, bbox_inches="tight")
    plt.close(f_d)
    print(f"Wrote {out_d}")

    # ---------- Sanity print ----------
    print("\nDistribution summary (per agent):")
    for label, t in trials.items():
        vfe = np.asarray(
            [x["vfe_value"] for x in t if x.get("vfe_value") is not None],
            dtype=float,
        )
        mt = np.asarray([x["mttr_seconds"] for x in t], dtype=float)
        cs = np.asarray(
            [x["certainty_score"]
                for x in t if x.get("certainty_score") is not None],
            dtype=float,
        )
        sc = np.asarray([x["repair_accuracy"] for x in t], dtype=float)
        print(f"  {label}")
        print(f"    VFE      : {fmt_quartiles(vfe)}")
        print(f"    MTTR (s) : {fmt_quartiles(mt)}")
        print(f"    cert     : {fmt_quartiles(cs)}")
        print(f"    score    : {fmt_quartiles(sc)}")

    print("\nAURORA VFE by fault class:")
    by_fault: dict[str, list[float]] = {}
    for t in trials["AURORA"]:
        f = t.get("fault_type")
        v = t.get("vfe_value")
        if f and v is not None:
            by_fault.setdefault(f, []).append(v)
    for f in ("network_drop", "cpu_spike", "memory_leak"):
        a = np.asarray(by_fault.get(f, []), dtype=float)
        print(f"  {f:<14}: {fmt_quartiles(a)}")


if __name__ == "__main__":
    main()
