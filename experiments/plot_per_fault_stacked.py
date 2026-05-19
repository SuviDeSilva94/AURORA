"""
Per-fault outcome composition stacked bar chart (publication-quality
redesign).

One stacked bar per (workload, fault_class) pair, with correct / abstain /
destructive proportions. Workload-group headers, destructive-value labels
on each bar that carries red mass, and a clean horizontal layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scienceplots  # noqa: F401


plt.style.use(["science", "ieee", "no-latex"])

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.9,
})


RESULTS = Path(__file__).resolve().parent / "results"


WORKLOADS = [
    ("CCTV (synthetic)",         "aurora_results.json",
     ["network_drop", "cpu_spike", "memory_leak"],
     "#dee9f5"),
    ("Motor (synthetic)",        "motor_aurora_results.json",
     ["bearing_wear", "thermal_overload", "coolant_loss"],
     "#dee9f5"),
    ("AI4I 2020 (real)",         "real_aurora_results.json",
     ["HDF", "PWF", "OSF"],
     "#fbe9d4"),
    ("Steel Plates (real)",      "steel_aurora_results.json",
     ["Pastry", "K_Scatch", "Stains"],
     "#fbe9d4"),
]


def load_per_fault(json_path: Path):
    trials = json.load(open(json_path))["trials"]
    out = {}
    for t in trials:
        f = t.get("fault_type", "?")
        c = out.setdefault(f, Counter())
        if t.get("destructive_action"):
            c["destructive"] += 1
        elif t.get("abstention"):
            c["abstained"] += 1
        elif t.get("action_outcome") == "correct":
            c["correct"] += 1
    return out


def main():
    fig, ax = plt.subplots(figsize=(13.5, 5.0))

    c_correct  = "#2ca02c"
    c_abstain  = "#ffbf2e"
    c_destruct = "#d62728"

    bar_w = 0.72
    group_gap = 0.9

    x = 0.0
    x_positions = []
    x_labels = []
    group_centres = []
    group_ranges = []

    for wi, (wname, fname, faults, band_color) in enumerate(WORKLOADS):
        per_fault = load_per_fault(RESULTS / fname)
        group_start = x
        for fault in faults:
            c = per_fault.get(fault, Counter())
            tot = sum(c.values()) or 1
            p_correct  = 100 * c.get("correct", 0)   / tot
            p_abstain  = 100 * c.get("abstained", 0) / tot
            p_destruct = 100 * c.get("destructive", 0) / tot

            ax.bar(x, p_correct,  bar_w, color=c_correct,
                   edgecolor="black", linewidth=0.4,
                   label="Correct" if (wi == 0 and fault == faults[0]) else None,
                   zorder=3)
            ax.bar(x, p_abstain,  bar_w, bottom=p_correct, color=c_abstain,
                   edgecolor="black", linewidth=0.4,
                   label="Abstain" if (wi == 0 and fault == faults[0]) else None,
                   zorder=3)
            ax.bar(x, p_destruct, bar_w, bottom=p_correct + p_abstain,
                   color=c_destruct, edgecolor="black", linewidth=0.4,
                   label="Destructive" if (wi == 0 and fault == faults[0]) else None,
                   zorder=3)
            # Destructive percentage label on top, only if nonzero
            if p_destruct > 0.01:
                ax.text(x, 102, f"{p_destruct:.2f}%", ha="center", va="bottom",
                        fontsize=8.5, color=c_destruct, fontweight="bold")
            x_positions.append(x)
            x_labels.append(fault.replace("_", "\\_"))
            x += 1
        group_centres.append((group_start + x - 1) / 2)
        group_ranges.append((group_start - 0.5, x - 0.5, band_color, wname))
        x += group_gap

    # ─── Background bands per workload ──────────────────────────────────
    for x_left, x_right, color, _ in group_ranges:
        band = mpatches.Rectangle(
            (x_left, 0), x_right - x_left, 100,
            facecolor=color, alpha=0.30, zorder=0,
        )
        ax.add_patch(band)

    # ─── Workload group headers ────────────────────────────────────────
    for center, (_, _, _, wname) in zip(group_centres, group_ranges):
        ax.annotate(wname, xy=(center, 116), xycoords="data",
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold",
                    annotation_clip=False)

    # ─── Per-fault tick labels ────────────────────────────────────────
    ax.set_xticks(x_positions)
    # Display: keep names readable (replace LaTeX \_ with _ for display)
    display_labels = [lbl.replace("\\_", "_") for lbl in x_labels]
    ax.set_xticklabels(display_labels, rotation=25, ha="right", fontsize=10)

    # ─── Y-axis ───────────────────────────────────────────────────────
    ax.set_ylabel("Outcome share (%)")
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Reference line at 100%
    ax.axhline(100, color="#555555", linewidth=0.6, linestyle=":", zorder=2)

    # ─── Legend below ─────────────────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True,
              framealpha=0.95, edgecolor="#888888")

    # ─── Title ────────────────────────────────────────────────────────
    ax.set_title("AURORA per-fault outcome composition across four workloads",
                 pad=28, fontsize=12.5, fontweight="bold")

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = RESULTS / "aurora_per_fault_stacked.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
