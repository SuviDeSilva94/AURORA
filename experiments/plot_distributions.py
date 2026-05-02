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

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"

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


def panel_violin(ax, data_per_agent, ylabel, title, ylim=None):
    """Generic violin-with-box-inside, one violin per agent."""
    labels = list(data_per_agent.keys())
    data = [np.asarray(data_per_agent[k], dtype=float) for k in labels]

    safe_data = []
    for arr in data:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            arr = np.array([0.0])
        safe_data.append(arr)

    parts = ax.violinplot(safe_data, showmeans=False, showmedians=False, showextrema=False)
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
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3, axis="y")


def panel_vfe_by_fault(ax, aurora_trials, title="(a) VFE distribution by fault class (AURORA)"):
    """AURORA only: VFE distribution split by injected fault class."""
    groups = {"network_drop": [], "cpu_spike": [], "memory_leak": []}
    for t in aurora_trials:
        f = t.get("fault_type")
        v = t.get("vfe_value")
        if f in groups and v is not None:
            groups[f].append(v)
    labels = ["network_drop", "cpu_spike", "memory_leak"]
    data = [np.asarray(groups[k], dtype=float) for k in labels]
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
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
    ax.axhline(3.85, color="red", ls="--", lw=1.2, label=r"$\mathcal{F}_{\rm th}=3.85$")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Network Drop", "CPU Spike", "Memory Leak"], fontsize=8)
    ax.set_ylabel(r"VFE score $\mathcal{F}_{\rm score}$")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)


def apply_linear_mttr(ax) -> None:
    """Linear y-axis for MTTR. Rule-Based clusters near 1 ms, AIF/AURORA near
    13 ms; the linear axis honestly shows the ~10x gap rather than hiding it
    under a log axis. Consistent with all other panels (linear scale)."""
    ax.set_ylim(-0.002, 0.05)
    ax.grid(True, alpha=0.3, axis="y")


def fmt_quartiles(a: np.ndarray) -> str:
    if a.size == 0:
        return "n/a"
    return (
        f"median={np.median(a):.4f}, "
        f"IQR=[{np.quantile(a, 0.25):.4f}, {np.quantile(a, 0.75):.4f}], "
        f"min={a.min():.4f}, max={a.max():.4f}"
    )


def main() -> None:
    trials = {label: load_trials(RESULTS_DIR / fname) for label, fname in AGENTS.items()}

    mttr = {k: [t["mttr_seconds"] for t in v] for k, v in trials.items()}
    cert = {
        k: [t["certainty_score"] for t in v if t.get("certainty_score") is not None]
        for k, v in trials.items()
    }
    score = {k: [t["repair_accuracy"] for t in v] for k, v in trials.items()}

    # ---------- Combined 2x2 master ----------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    panel_vfe_by_fault(axes[0, 0], trials["AURORA"])

    panel_violin(axes[0, 1], mttr,
                 ylabel="MTTR (s)",
                 title="(b) Decision latency per agent")
    apply_linear_mttr(axes[0, 1])

    panel_violin(axes[1, 0], cert,
                 ylabel=r"Posterior certainty $P_{\max}$",
                 title="(c) Posterior certainty per agent",
                 ylim=(0, 1.05))
    axes[1, 0].axhline(0.70, color="blue", ls="--", lw=1.2, alpha=0.7)
    axes[1, 0].text(0.55, 0.72, r"$\tau=0.70$", color="blue", fontsize=8)

    panel_violin(axes[1, 1], score,
                 ylabel=r"Per-trial score $S_i$",
                 title="(d) Per-trial outcome score per agent",
                 ylim=(-0.05, 1.05))

    fig.tight_layout()
    out = RESULTS_DIR / "distributions.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # ---------- Per-panel standalone PDFs ----------
    # (a) VFE by fault class
    f_a, ax_a = plt.subplots(figsize=(5.0, 3.5))
    panel_vfe_by_fault(ax_a, trials["AURORA"], title="VFE distribution by fault class (AURORA)")
    f_a.tight_layout()
    out_a = RESULTS_DIR / "dist_vfe_by_fault.pdf"
    f_a.savefig(out_a, bbox_inches="tight"); plt.close(f_a)
    print(f"Wrote {out_a}")

    # (b) MTTR per agent (LOG y)
    f_b, ax_b = plt.subplots(figsize=(5.0, 3.5))
    panel_violin(ax_b, mttr,
                 ylabel="MTTR (s)",
                 title="Decision latency per agent")
    apply_linear_mttr(ax_b)
    f_b.tight_layout()
    out_b = RESULTS_DIR / "dist_mttr.pdf"
    f_b.savefig(out_b, bbox_inches="tight"); plt.close(f_b)
    print(f"Wrote {out_b}")

    # (c) Posterior certainty per agent
    f_c, ax_c = plt.subplots(figsize=(5.0, 3.5))
    panel_violin(ax_c, cert,
                 ylabel=r"Posterior certainty $P_{\max}$",
                 title="Posterior certainty per agent",
                 ylim=(0, 1.05))
    ax_c.axhline(0.70, color="blue", ls="--", lw=1.2, alpha=0.7)
    ax_c.text(0.55, 0.72, r"$\tau=0.70$", color="blue", fontsize=8)
    f_c.tight_layout()
    out_c = RESULTS_DIR / "dist_certainty.pdf"
    f_c.savefig(out_c, bbox_inches="tight"); plt.close(f_c)
    print(f"Wrote {out_c}")

    # (d) Per-trial score per agent
    f_d, ax_d = plt.subplots(figsize=(5.0, 3.5))
    panel_violin(ax_d, score,
                 ylabel=r"Per-trial score $S_i$",
                 title="Per-trial outcome score per agent",
                 ylim=(-0.05, 1.05))
    f_d.tight_layout()
    out_d = RESULTS_DIR / "dist_score.pdf"
    f_d.savefig(out_d, bbox_inches="tight"); plt.close(f_d)
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
            [x["certainty_score"] for x in t if x.get("certainty_score") is not None],
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
