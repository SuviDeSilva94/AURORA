"""
Gate-1 firing visualization for the multi-fault experiment.

Reads ``experiments/results/aurora_multi_fault_diagnostics.json`` and produces:

  - dist_pmax_by_anomaly.pdf : Pmax distribution per |At| (validates the
    flattening assumption behind Eq. 11).
  - decision_space_gate1.pdf : (Pmax, F) scatter, marker shape encodes |At|,
    horizontal lines mark per-|At| dynamic τ. Highlights trials in
    [τ_dynamic(n), τ_base) — the "rescued" trials where the dynamic schedule
    converted an abstention into a correct execution.
  - gate_firing_by_anomaly.pdf : stacked bars per |At| showing how often
    Gate 1 / Gate 2 / both / neither fire, plus rescue count under the
    dynamic schedule vs a hypothetical fixed τ.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DIAG_PATH = RESULTS_DIR / "aurora_multi_fault_diagnostics.json"

ANOMALY_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}
ANOMALY_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}


def load_diagnostics() -> list[dict]:
    with open(DIAG_PATH) as f:
        return json.load(f)


def plot_pmax_distribution(diags: list[dict]) -> None:
    by_n: dict[int, list[float]] = defaultdict(list)
    for d in diags:
        n = d.get("anomaly_count")
        c = d.get("certainty")
        if n is not None and c is not None:
            by_n[n].append(c)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ns = sorted(by_n)
    data = [by_n[n] for n in ns]
    parts = ax.violinplot(data, positions=ns, widths=0.7,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, n in zip(parts["bodies"], ns):
        body.set_facecolor(ANOMALY_COLORS.get(n, "gray"))
        body.set_alpha(0.55)
        body.set_edgecolor("black")
    ax.boxplot(data, positions=ns, widths=0.18, patch_artist=False,
               medianprops={"color": "black", "linewidth": 1.3},
               flierprops={"marker": "x", "markersize": 3, "alpha": 0.6})

    if diags:
        tau_base = diags[0].get("tau_base", 0.70)
        ax.axhline(tau_base, color="blue", ls="--", lw=1.2, alpha=0.8,
                   label=fr"$\tau_{{\mathrm{{base}}}}={tau_base:.2f}$")

    ax.set_xticks(ns)
    ax.set_xlabel(r"Anomaly count $|A_t|$")
    ax.set_ylabel(r"Posterior certainty $P_{\max}$")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)

    out = RESULTS_DIR / "dist_pmax_by_anomaly.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_decision_space(diags: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    tau_base = diags[0].get("tau_base", 0.70) if diags else 0.70
    f_th = diags[0].get("vfe_threshold", 3.85) if diags else 3.85

    by_n: dict[int, list[dict]] = defaultdict(list)
    for d in diags:
        n = d.get("anomaly_count")
        if n is not None and d.get("certainty") is not None and d.get("vfe") is not None:
            by_n[n].append(d)

    for n in sorted(by_n):
        rows = by_n[n]
        marker = ANOMALY_MARKERS.get(n, "o")
        color = ANOMALY_COLORS.get(n, "gray")
        x = [r["certainty"] for r in rows]
        y = [r["vfe"] for r in rows]
        ax.scatter(x, y, marker=marker, c=color, alpha=0.55, s=28,
                   edgecolors="black", linewidths=0.4,
                   label=fr"$|A_t|={n}$ (n={len(rows)})")

        # Per-|At| dynamic τ horizontal projection (vertical line at τ_eff)
        tau_eff = rows[0].get("effective_tau")
        if tau_eff is not None and tau_eff < tau_base:
            ax.axvline(tau_eff, color=color, ls=":", lw=1.0, alpha=0.7)

    # Highlight rescued trials: cert in [τ_eff(n), τ_base) AND F < F_th
    rescue_x, rescue_y = [], []
    for n in sorted(by_n):
        rows = by_n[n]
        tau_eff = rows[0].get("effective_tau", tau_base)
        for r in rows:
            c, v = r["certainty"], r["vfe"]
            if tau_eff <= c < tau_base and v < f_th:
                rescue_x.append(c)
                rescue_y.append(v)
    if rescue_x:
        ax.scatter(rescue_x, rescue_y, marker="o", facecolors="none",
                   edgecolors="red", s=120, linewidths=1.4,
                   label=f"rescued by dynamic τ (n={len(rescue_x)})")

    ax.axhline(f_th, color="red", ls="--", lw=1.2,
               label=fr"$\mathcal{{F}}_{{\mathrm{{th}}}}={f_th:.2f}$")
    ax.axvline(tau_base, color="blue", ls="--", lw=1.2,
               label=fr"$\tau_{{\mathrm{{base}}}}={tau_base:.2f}$")

    ax.set_xlabel(r"Posterior certainty $P_{\max}$")
    ax.set_ylabel(r"VFE $\mathcal{F}$")
    ax.set_xlim(-0.02, 1.05)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    out = RESULTS_DIR / "decision_space_gate1.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_gate_firing(diags: list[dict]) -> None:
    by_n: dict[int, list[dict]] = defaultdict(list)
    for d in diags:
        n = d.get("anomaly_count")
        if n is not None:
            by_n[n].append(d)

    ns = sorted(by_n)
    g1_only = []
    g2_only = []
    both = []
    neither = []
    rescued = []
    would_fire_fixed = []  # would have fired Gate 1 under fixed τ_base

    tau_base = diags[0].get("tau_base", 0.70) if diags else 0.70
    f_th = diags[0].get("vfe_threshold", 3.85) if diags else 3.85

    for n in ns:
        rows = by_n[n]
        tau_eff = rows[0].get("effective_tau", tau_base)
        a = b = c = d = r = wf = 0
        for row in rows:
            cert = row["certainty"]
            vfe = row["vfe"]
            f1 = bool(row.get("gate1_fired"))
            f2 = bool(row.get("gate2_fired"))
            if f1 and f2: b += 1
            elif f1: a += 1
            elif f2: c += 1
            else: d += 1
            if cert < tau_base:
                wf += 1
            if tau_eff <= cert < tau_base and vfe < f_th:
                r += 1
        g1_only.append(a)
        both.append(b)
        g2_only.append(c)
        neither.append(d)
        rescued.append(r)
        would_fire_fixed.append(wf)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    x = np.arange(len(ns))
    width = 0.65
    ax1.bar(x, neither, width, label="executed", color="#2ca02c", alpha=0.85)
    ax1.bar(x, g2_only, width, bottom=neither, label="Gate 2 only", color="#ff7f0e", alpha=0.85)
    bottom2 = [a + b for a, b in zip(neither, g2_only)]
    ax1.bar(x, g1_only, width, bottom=bottom2, label="Gate 1 only", color="#1f77b4", alpha=0.85)
    bottom3 = [a + b for a, b in zip(bottom2, g1_only)]
    ax1.bar(x, both, width, bottom=bottom3, label="both gates", color="#d62728", alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in ns])
    ax1.set_xlabel(r"Anomaly count $|A_t|$")
    ax1.set_ylabel("Trial count")
    ax1.set_title("Gate firing breakdown")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(alpha=0.2, axis="y")

    fired_fixed = [wf for wf in would_fire_fixed]
    fired_dynamic = [a + b for a, b in zip(g1_only, both)]
    width2 = 0.38
    ax2.bar(x - width2/2, fired_fixed, width2,
            label=fr"fixed $\tau={tau_base:.2f}$", color="#444444", alpha=0.85)
    ax2.bar(x + width2/2, fired_dynamic, width2,
            label="dynamic τ (Eq. 11)", color="#1f77b4", alpha=0.85)
    for i, r in enumerate(rescued):
        if r > 0:
            ax2.annotate(f"+{r} rescued",
                         xy=(i + width2/2, fired_dynamic[i]),
                         xytext=(i + width2/2, fired_dynamic[i] + 4),
                         ha="center", fontsize=9, color="red")

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in ns])
    ax2.set_xlabel(r"Anomaly count $|A_t|$")
    ax2.set_ylabel("Gate-1 firings")
    ax2.set_title("Dynamic τ vs fixed τ — Gate 1")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(alpha=0.2, axis="y")

    out = RESULTS_DIR / "gate_firing_by_anomaly.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def print_summary(diags: list[dict]) -> None:
    by_n: dict[int, list[dict]] = defaultdict(list)
    for d in diags:
        n = d.get("anomaly_count")
        if n is not None:
            by_n[n].append(d)

    print()
    print(f"{'|At|':<5} {'trials':<7} {'P̃max med':<11} {'τ_eff':<6} "
          f"{'fixed-τ G1':<11} {'dyn-τ G1':<10} {'rescued':<8} {'G2':<6}")
    tau_base = diags[0].get("tau_base", 0.70) if diags else 0.70
    f_th = diags[0].get("vfe_threshold", 3.85) if diags else 3.85
    for n in sorted(by_n):
        rows = by_n[n]
        certs = [r["certainty"] for r in rows]
        tau_eff = rows[0].get("effective_tau", tau_base)
        fixed = sum(1 for c in certs if c < tau_base)
        dyn = sum(1 for r in rows if r.get("gate1_fired"))
        rescued = sum(
            1 for r in rows
            if tau_eff <= r["certainty"] < tau_base and r["vfe"] < f_th
        )
        g2 = sum(1 for r in rows if r.get("gate2_fired"))
        print(
            f"{n:<5} {len(rows):<7} {np.median(certs):<11.3f} {tau_eff:<6.2f} "
            f"{fixed:<11} {dyn:<10} {rescued:<8} {g2:<6}"
        )


def main() -> None:
    diags = load_diagnostics()
    print(f"Loaded {len(diags)} multi-fault trials")
    print_summary(diags)
    plot_pmax_distribution(diags)
    plot_decision_space(diags)
    plot_gate_firing(diags)


if __name__ == "__main__":
    main()
