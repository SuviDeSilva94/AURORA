"""
Results Analysis and Visualization

Loads experimental results and generates:
- Comparison tables
- Bar charts (repair accuracy, MTTR, abstention rate, destructive actions)
- Statistical significance tests
- Publication-ready figures for thesis
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import Dict, List, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
from loguru import logger

from src.evaluation.proportion_ci import (
    bootstrap_mean_ci,
    wilson_interval,
    wilson_summary_for_agent_metrics,
)

import scienceplots

# Apply IEEE formatting with SciencePlots
plt.style.use(["science", "ieee", "no-latex"])

colors = ['#0C5DA5', '#00B945', '#FF9500', '#845B97', '#474747', '#9E9E9E']
# Aggressive font sizing for IEEE two-column print: figures are scaled to a
# single column (~3.5"), so on-figure text needs to be visibly larger than the
# caption (which renders at ~8pt) once scaled. These overrides intentionally
# overwrite the SciencePlots IEEE preset (which uses ~7pt fonts).
plt.rcParams.update({
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


class ResultsAnalyzer:
    """
    Analyzes experimental results and generates visualizations
    """
    
    def __init__(self, results_dir: str = "experiments/results"):
        self.results_dir = Path(results_dir)
        self.results = {}
        self.load_results()
    
    def load_results(self):
        """Load all result files"""
        summary_file = self.results_dir / "comparison_summary.json"
        
        if not summary_file.exists():
            logger.error(f"Results file not found: {summary_file}")
            logger.error("Run experiments first: python experiments/run_comparison.py")
            return
        
        with open(summary_file, 'r') as f:
            self.results = json.load(f)
        
        logger.info(f"Loaded results for {len(self.results)} agents")
    
    def _trial_repair_accuracies(self, agent_key: str) -> List[float]:
        """Load per-trial repair_accuracy for bootstrap CI on the continuous score."""
        path = self.results_dir / f"{agent_key}_results.json"
        if not path.exists():
            return []
        with open(path) as f:
            payload = json.load(f)
        return [float(t["repair_accuracy"]) for t in payload.get("trials", [])]

    def compute_all_confidence_intervals(
        self, confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        Wilson CIs on binomial counts (correct / abstain / destructive) and
        bootstrap CI on mean trial-level repair_accuracy (continuous score).
        """
        out: Dict[str, Any] = {"confidence": confidence, "agents": {}}
        for agent_name, metrics in self.results.items():
            wilson = wilson_summary_for_agent_metrics(metrics, confidence=confidence)
            accs = self._trial_repair_accuracies(agent_name)
            boot_lo, boot_hi = bootstrap_mean_ci(accs, confidence=confidence)
            out["agents"][agent_name] = {
                "wilson": wilson,
                "bootstrap_mean_repair_accuracy": {
                    "mean": metrics["mean_repair_accuracy"],
                    "ci_low": boot_lo,
                    "ci_high": boot_hi,
                    "n_trials": len(accs),
                },
            }
        return out

    def save_confidence_intervals_json(
        self, path: Optional[Path] = None, confidence: float = 0.95
    ) -> Path:
        path = path or (self.results_dir / "confidence_intervals.json")
        data = self.compute_all_confidence_intervals(confidence=confidence)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved Wilson + bootstrap CIs to {path}")
        return path

    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create comparison table for thesis (includes Wilson 95% intervals on proportions).
        """
        data = []
        agent_keys = sorted(
            self.results.keys(),
            key=lambda k: self.results[k]["mean_repair_accuracy"],
            reverse=True,
        )

        for agent_name in agent_keys:
            metrics = self.results[agent_name]
            w = wilson_summary_for_agent_metrics(metrics, confidence=0.95)
            rc = w["repair_correct"]
            ab = w["abstention"]
            de = w["destructive"]

            def _fmt_pct_interval(d: Dict) -> str:
                return (
                    f"{d['p_hat']:.1%} "
                    f"[{d['ci_low']:.1%}, {d['ci_high']:.1%}]"
                )

            data.append(
                {
                    "Agent Type": self._format_agent_name(agent_name),
                    "Repair Accuracy (mean trial score)": f"{metrics['mean_repair_accuracy']:.1%}",
                    "Repair correct (Wilson 95%)": _fmt_pct_interval(rc),
                    "MTTR (seconds)": f"{metrics['mean_mttr_seconds']:.2f}",
                    "Resolution Rate": f"{metrics.get('resolution_rate', 0):.1%}",
                    "Abstention (Wilson 95%)": _fmt_pct_interval(ab),
                    "Destructive (Wilson 95%)": _fmt_pct_interval(de),
                    "Correct Actions": metrics["num_correct"],
                    "Incorrect Actions": metrics["num_incorrect"],
                    "Trials": metrics["num_trials"],
                }
            )

        return pd.DataFrame(data)
    
    def plot_comparison_chart(self):
        """
        Create publication-quality comparison chart
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # Suptitle removed — caption is rendered by the LaTeX figure environment.
        
        # Prepare data
        agent_keys = list(self.results.keys())
        agents = [self._format_agent_name(name) for name in agent_keys]

        # Metric 1: Repair correct rate with Wilson 95% error bars (binomial)
        p_cor = [
            self.results[k]["num_correct"] / max(self.results[k]["num_trials"], 1)
            for k in agent_keys
        ]
        w_lo = [
            wilson_interval(self.results[k]["num_correct"], self.results[k]["num_trials"])[0]
            for k in agent_keys
        ]
        w_hi = [
            wilson_interval(self.results[k]["num_correct"], self.results[k]["num_trials"])[1]
            for k in agent_keys
        ]
        y = [100.0 * p for p in p_cor]
        yerr = [
            [100.0 * max(0.0, p - lo), 100.0 * max(0.0, hi - p)]
            for p, lo, hi in zip(p_cor, w_lo, w_hi)
        ]
        yerr_arr = np.array(yerr).T
        ax1 = axes[0, 0]
        x_pos = np.arange(len(agents))
        colors = ["#ff9999", "#99ccff", "#66b266"]
        bars1 = ax1.bar(
            x_pos,
            y,
            color=colors[: len(agents)],
            yerr=yerr_arr,
            capsize=4,
            error_kw={"linewidth": 1.5, "ecolor": "#333333"},
        )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(agents)
        ax1.set_ylabel('Repair correct rate (%)', fontweight='bold')
        # Subplot title removed — caption is rendered by the LaTeX figure environment.
        ax1.set_ylim(0, 100)
        ax1.axhline(y=80, color='red', linestyle='--', alpha=0.3, label='Target (80%)')
        for bar, pct in zip(bars1, y):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )
        ax1.legend()
        
        # Metric 2: MTTR
        mttrs = [self.results[k]["mean_mttr_seconds"] for k in agent_keys]
        ax2 = axes[0, 1]
        bars2 = ax2.bar(
            x_pos,
            mttrs,
            color=colors[: len(agents)],
        )
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(agents)
        ax2.set_ylabel('Mean Time to Repair (seconds)', fontweight='bold')
        # Subplot title removed — caption is rendered by the LaTeX figure environment.
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # Metric 3: Abstention with Wilson CIs
        abst_y = []
        abst_err = []
        for k in agent_keys:
            m = self.results[k]
            kk, nn = m["num_abstained"], m["num_trials"]
            p = kk / max(nn, 1)
            lo, hi = wilson_interval(kk, nn)
            abst_y.append(100.0 * p)
            abst_err.append([100.0 * max(0.0, p - lo), 100.0 * max(0.0, hi - p)])
        abst_err_arr = np.array(abst_err).T
        ax3 = axes[1, 0]
        bars3 = ax3.bar(
            x_pos,
            abst_y,
            color=colors[: len(agents)],
            yerr=abst_err_arr,
            capsize=4,
            error_kw={"linewidth": 1.5, "ecolor": "#333333"},
        )
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(agents)
        ax3.set_ylabel('Abstention Rate (%)', fontweight='bold')
        # Subplot title removed — caption is rendered by the LaTeX figure environment.
        ymax = max(abst_y) if abst_y else 10.0
        ax3.set_ylim(0, max(ymax * 1.35, 10.0))
        for bar, pct in zip(bars3, abst_y):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )
        
        # Metric 4: Destructive rate with Wilson CIs
        dest_y = []
        dest_err = []
        for k in agent_keys:
            m = self.results[k]
            kk, nn = m["num_destructive"], m["num_trials"]
            p = kk / max(nn, 1)
            lo, hi = wilson_interval(kk, nn)
            dest_y.append(100.0 * p)
            dest_err.append([100.0 * max(0.0, p - lo), 100.0 * max(0.0, hi - p)])
        dest_err_arr = np.array(dest_err).T
        ax4 = axes[1, 1]
        bars4 = ax4.bar(
            x_pos,
            dest_y,
            color=colors[: len(agents)],
            yerr=dest_err_arr,
            capsize=4,
            error_kw={"linewidth": 1.5, "ecolor": "#333333"},
        )
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(agents)
        ax4.set_ylabel('Destructive Action Rate (%)', fontweight='bold')
        # Subplot title removed — caption is rendered by the LaTeX figure environment.
        dmax = max(dest_y) if dest_y else 0.0
        ax4.set_ylim(0, max(dmax * 1.35, 10.0))
        for bar, pct in zip(bars4, dest_y):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )
        
        plt.tight_layout()
        
        # Save figure (PDF for IEEE vector compliance)
        output_file = self.results_dir / "comparison_chart.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison chart to {output_file}")
        
        plt.show()
    
    def plot_by_fault_type(self):
        """
        Plot performance breakdown by fault type as TWO separate PDFs so the
        LaTeX side can lay them out as subfigures (a) and (b) with captions
        rendered by the document, not embedded in the figure.
        """
        fault_data = []
        for agent_name, metrics in self.results.items():
            for fault_type, fault_metrics in metrics['metrics_by_fault'].items():
                fault_data.append({
                    'Agent': self._format_agent_name(agent_name),
                    'Fault Type': self._format_fault_name(fault_type),
                    'Accuracy': fault_metrics['mean_accuracy'] * 100,
                    'MTTR': fault_metrics['mean_mttr']
                })
        df = pd.DataFrame(fault_data)

        # Common legend styling: one horizontal row inside the axes, no
        # "Agent Type" title (saves a row), short shorthand labels, and a
        # semi-transparent white background so bars remain visible behind.
        # ncol=3 collapses the legend from a 4-row block into a single
        # thin strip that fits in the headroom above the bars.
        short_label = {
            'AIF (No Gate)':       'AIF',
            'AURORA (Proposed)':   'AURORA',
            'Rule-Based':          'Rule-Based',
        }
        legend_common = dict(
            ncol=3, columnspacing=1.2,
            framealpha=0.85, facecolor='white', edgecolor='0.4',
            fancybox=True, borderpad=0.35,
            handletextpad=0.4, handlelength=1.5,
        )

        def _apply_short_labels(ax):
            handles, labels = ax.get_legend_handles_labels()
            labels = [short_label.get(l, l) for l in labels]
            return handles, labels

        # ---- (a) Accuracy by fault type ----
        # 100% bars on Memory/Network Drop reach the top, so we lift the
        # ceiling to 118 to give a single legend row clean headroom across
        # the top of the plot.
        fig_a, ax_a = plt.subplots(figsize=(6.0, 3.5))
        pivot_acc = df.pivot(index='Fault Type', columns='Agent', values='Accuracy')
        pivot_acc.plot(kind='bar', ax=ax_a, rot=0)
        ax_a.set_xlabel("")
        ax_a.set_ylabel('Repair Accuracy (%)')
        h, l = _apply_short_labels(ax_a)
        ax_a.legend(h, l, loc='upper center', **legend_common)
        ax_a.set_ylim(0, 118)
        fig_a.tight_layout()
        out_a = self.results_dir / "fault_accuracy.pdf"
        fig_a.savefig(out_a, dpi=300, bbox_inches='tight')
        plt.close(fig_a)
        logger.info(f"Saved fault accuracy plot to {out_a}")

        # ---- (b) MTTR by fault type ----
        # AIF and AURORA bars cluster at ~0.013s. ymax * 1.20 gives one
        # legend-row of headroom across the top without the legend
        # dominating the figure as a tall sidebar.
        fig_b, ax_b = plt.subplots(figsize=(6.0, 3.5))
        pivot_mttr = df.pivot(index='Fault Type', columns='Agent', values='MTTR')
        pivot_mttr.plot(kind='bar', ax=ax_b, rot=0)
        ax_b.set_xlabel("")
        ax_b.set_ylabel('MTTR (seconds)')
        h, l = _apply_short_labels(ax_b)
        ax_b.legend(h, l, loc='upper center', **legend_common)
        ymax = float(pivot_mttr.values.max())
        ax_b.set_ylim(0, ymax * 1.20)
        fig_b.tight_layout()
        out_b = self.results_dir / "fault_mttr.pdf"
        fig_b.savefig(out_b, dpi=300, bbox_inches='tight')
        plt.close(fig_b)
        logger.info(f"Saved fault MTTR plot to {out_b}")

        # Backwards-compat: combined figure (no suptitle, no per-panel titles).
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        pivot_acc.plot(kind='bar', ax=axes[0], rot=0)
        axes[0].set_xlabel("")
        axes[0].set_ylabel('Repair Accuracy (%)')
        h, l = _apply_short_labels(axes[0])
        axes[0].legend(h, l, loc='upper center', **legend_common)
        axes[0].set_ylim(0, 118)
        axes[0].grid(True, alpha=0.3, axis='y')
        pivot_mttr.plot(kind='bar', ax=axes[1], rot=0)
        axes[1].set_xlabel("")
        axes[1].set_ylabel('MTTR (seconds)')
        h, l = _apply_short_labels(axes[1])
        axes[1].legend(h, l, loc='upper center', **legend_common)
        axes[1].set_ylim(0, ymax * 1.20)
        axes[1].grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        output_file = self.results_dir / "fault_type_breakdown.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved combined fault type breakdown to {output_file}")
    
    def statistical_significance_test(self):
        """
        Perform t-tests to determine statistical significance
        
        Compare AURORA vs each baseline
        """
        logger.info("\n" + "="*70)
        logger.info("STATISTICAL SIGNIFICANCE TESTS")
        logger.info("="*70)
        
        # Load individual trial data
        aurora_file = self.results_dir / "aurora_results.json"
        rule_file = self.results_dir / "rule_based_results.json"
        aif_file = self.results_dir / "aif_no_gate_results.json"
        
        if not all([aurora_file.exists(), rule_file.exists(), aif_file.exists()]):
            logger.warning("Individual trial files not found, skipping significance tests")
            return
        
        with open(aurora_file) as f:
            aurora_trials = json.load(f)['trials']
        with open(rule_file) as f:
            rule_trials = json.load(f)['trials']
        with open(aif_file) as f:
            aif_trials = json.load(f)['trials']
        
        # Extract repair accuracies
        aurora_acc = [t['repair_accuracy'] for t in aurora_trials]
        rule_acc = [t['repair_accuracy'] for t in rule_trials]
        aif_acc = [t['repair_accuracy'] for t in aif_trials]
        
        # T-tests
        t_stat_rule, p_value_rule = stats.ttest_ind(aurora_acc, rule_acc)
        t_stat_aif, p_value_aif = stats.ttest_ind(aurora_acc, aif_acc)
        
        print(f"\nAURORA vs Rule-Based:")
        print(f"  t-statistic: {t_stat_rule:.4f}")
        print(f"  p-value: {p_value_rule:.4f}")
        print(f"  Significant at α=0.05: {'YES ✓' if p_value_rule < 0.05 else 'NO ✗'}")
        
        print(f"\nAURORA vs AIF-No-Gate:")
        print(f"  t-statistic: {t_stat_aif:.4f}")
        print(f"  p-value: {p_value_aif:.4f}")
        print(f"  Significant at α=0.05: {'YES ✓' if p_value_aif < 0.05 else 'NO ✗'}")
        
        print("\n" + "="*70)
    
    def generate_thesis_summary(self):
        """
        Generate a text summary suitable for thesis
        """
        summary_file = self.results_dir / "thesis_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EXPERIMENTAL RESULTS SUMMARY (For Thesis Chapter 4)\n")
            f.write("="*70 + "\n\n")
            
            ci_all = self.compute_all_confidence_intervals(confidence=0.95)
            for agent_name, metrics in self.results.items():
                formatted_name = self._format_agent_name(agent_name)
                w = ci_all["agents"][agent_name]["wilson"]
                rc, ab, de = w["repair_correct"], w["abstention"], w["destructive"]
                boot = ci_all["agents"][agent_name]["bootstrap_mean_repair_accuracy"]
                f.write(f"{formatted_name}:\n")
                f.write(
                    f"  • Repair Accuracy (mean trial score): {metrics['mean_repair_accuracy']:.1%} "
                    f"(±{metrics['std_repair_accuracy']:.1%}); "
                    f"bootstrap 95% CI [{boot['ci_low']:.1%}, {boot['ci_high']:.1%}]\n"
                )
                f.write(
                    f"  • Repair correct (binary): {rc['p_hat']:.1%} Wilson 95% "
                    f"[{rc['ci_low']:.1%}, {rc['ci_high']:.1%}] "
                    f"({rc['k']}/{rc['n']} trials)\n"
                )
                f.write(f"  • Mean Time to Repair: {metrics['mean_mttr_seconds']:.2f}s "
                       f"(±{metrics['std_mttr_seconds']:.2f}s)\n")
                f.write(f"  • Resolution Rate: {metrics.get('resolution_rate', 0):.1%}\n")
                f.write(
                    f"  • Abstention: {ab['p_hat']:.1%} Wilson 95% "
                    f"[{ab['ci_low']:.1%}, {ab['ci_high']:.1%}]\n"
                )
                f.write(
                    f"  • Destructive: {de['p_hat']:.1%} Wilson 95% "
                    f"[{de['ci_low']:.1%}, {de['ci_high']:.1%}]\n"
                )
                f.write(f"  • Correct Actions: {metrics['num_correct']}/{metrics['num_trials']}\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("KEY FINDINGS:\n")
            f.write("="*70 + "\n\n")
            
            # Calculate improvements
            aurora = self.results.get('aurora', {})
            rule = self.results.get('rule_based', {})
            aif = self.results.get('aif_no_gate', {})
            
            if aurora and rule:
                acc_improve = (aurora['mean_repair_accuracy'] - rule['mean_repair_accuracy']) * 100
                f.write(f"1. AURORA achieves {acc_improve:.1f} percentage points higher repair "
                       f"accuracy than rule-based baseline\n\n")
            
            if aurora and aif:
                dest_reduce = aif['destructive_action_rate'] - aurora['destructive_action_rate']
                f.write(f"2. AURORA reduces destructive actions by {dest_reduce:.1f} percentage "
                       f"points compared to AIF without gating\n\n")
            
            if aurora:
                f.write(f"3. AURORA demonstrates abstention capability ({aurora['abstention_rate']:.1f}%), "
                       f"refusing to act under uncertainty\n\n")
            
            f.write("="*70 + "\n")
        
        logger.info(f"Saved thesis summary to {summary_file}")
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_diagnostics(self, agent_key: str) -> List[Dict]:
        path = self.results_dir / f"{agent_key}_trial_diagnostics.json"
        if not path.exists():
            logger.warning(f"Diagnostics file not found: {path}")
            return []
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # New figures
    # ------------------------------------------------------------------

    def plot_vfe_distribution(self):
        """
        Fig: VFE Score Distribution by Fault Type (AURORA only).

        Histogram + KDE for each fault class with the VFE gate threshold
        drawn as a vertical dashed line.  Shows WHY abstention occurs and
        validates the threshold choice.
        """
        diag = self._load_diagnostics("aurora")
        if not diag:
            return

        fault_colors = {
            "network_drop": "#2196F3",
            "cpu_spike":    "#FF5722",
            "memory_leak":  "#4CAF50",
        }
        fault_labels = {
            "network_drop": "Network Drop",
            "cpu_spike":    "CPU Spike",
            "memory_leak":  "Memory Leak",
        }
        vfe_gate = 3.85

        fig, ax = plt.subplots(figsize=(10, 5))

        for fault, color in fault_colors.items():
            vfes = [t["vfe"] for t in diag if t["fault_type"] == fault]
            ax.hist(
                vfes,
                bins=30,
                alpha=0.45,
                color=color,
                label=fault_labels[fault],
                edgecolor="white",
                linewidth=0.4,
            )
            # KDE overlay
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vfes, bw_method=0.3)
            xs = np.linspace(min(vfes) - 0.2, max(vfes) + 0.2, 300)
            scale = len(vfes) * (max(vfes) - min(vfes)) / 30
            ax.plot(xs, kde(xs) * scale, color=color, linewidth=2)

        ax.axvline(x=vfe_gate, color="crimson", linestyle="--", linewidth=1.8,
                   label=f"VFE gate (F = {vfe_gate})")
        ax.set_xlabel("Variational Free Energy (F)", fontweight="bold")
        ax.set_ylabel("Trial Count", fontweight="bold")
        # Title removed — caption is rendered by the LaTeX figure environment.
        ax.legend(
            framealpha=0.85, facecolor="white", edgecolor="0.7",
            fancybox=True, borderpad=0.4,
        )

        # Annotate gate regions
        ymax = ax.get_ylim()[1]
        ax.text(vfe_gate - 0.15, ymax * 0.92, "Execute\n(F < gate)",
                ha="right", color="green", fontsize=9, fontstyle="italic")
        ax.text(vfe_gate + 0.15, ymax * 0.92, "Abstain\n(F ≥ gate)",
                ha="left", color="crimson", fontsize=9, fontstyle="italic")

        plt.tight_layout()
        out = self.results_dir / "vfe_distribution.pdf"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved VFE distribution to {out}")
        plt.show()

    def plot_safety_gate_scatter(self):
        """
        Fig: Safety Gate Decision Space — VFE vs Certainty (AURORA).

        One point per trial, coloured by outcome (correct / abstained).
        Gate boundaries drawn as dashed lines, quadrants labelled.
        Shows that the VFE gate is the operative safety mechanism.
        """
        diag = self._load_diagnostics("aurora")
        if not diag:
            return

        vfe_gate      = 3.85
        cert_gate     = 0.70
        outcome_style = {
            "correct":  {"color": "#2e7d32", "marker": "o", "label": "Correct execution"},
            "abstained": {"color": "#e65100", "marker": "^", "label": "Abstained (high VFE)"},
        }

        fig, ax = plt.subplots(figsize=(7.0, 4.5))

        rng = np.random.default_rng(42)

        for outcome, style in outcome_style.items():
            xs = np.array([t["certainty"] for t in diag if t["outcome"] == outcome])
            ys = np.array([t["vfe"]       for t in diag if t["outcome"] == outcome])
            # Add small jitter so overlapping points are visible
            xs = xs + rng.normal(0, 0.004, size=len(xs))
            ys = ys + rng.normal(0, 0.018, size=len(ys))
            ax.scatter(xs, ys, c=style["color"], marker=style["marker"],
                       alpha=0.45, s=22, label=style["label"], edgecolors="none")

        ax.axhline(y=vfe_gate,  color="crimson",   linestyle="--", linewidth=1.6,
                   label=f"VFE gate  F = {vfe_gate}")
        ax.axvline(x=cert_gate, color="steelblue", linestyle="--", linewidth=1.6,
                   label=f"Certainty gate  τ = {cert_gate}")

        # Quadrant annotations — placed in corners away from dense clusters
        ax.text(0.075, vfe_gate - 0.15, "Safe Zone  (execute)",
                ha="left", va="top", fontsize=10, color="#2e7d32", fontstyle="italic",
                transform=ax.get_yaxis_transform())
        ax.text(0.075, vfe_gate + 0.15, "High Uncertainty  (abstain)",
                ha="left", va="bottom", fontsize=10, color="crimson", fontstyle="italic",
                transform=ax.get_yaxis_transform())

        ax.set_xlabel(r"Posterior certainty $P_{max}$")
        ax.set_ylabel(r"VFE $\mathcal{F}$")
        # Title removed — caption is rendered by the LaTeX figure environment.
        ax.legend(
            framealpha=0.85, facecolor="white", edgecolor="0.7",
            fancybox=True, borderpad=0.4,
        )

        plt.tight_layout()
        out = self.results_dir / "safety_gate_scatter.pdf"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved safety gate scatter to {out}")
        plt.show()

    def plot_outcome_composition(self):
        """
        Fig: Per-Fault Outcome Composition — stacked bars for each
        agent × fault type showing correct / abstained / destructive /
        incorrect counts.  More informative than aggregated accuracy alone.
        """
        agent_keys  = ["rule_based", "aif_no_gate", "aurora"]
        fault_types = ["network_drop", "cpu_spike", "memory_leak"]
        outcome_cols = {
            "correct":     "#4CAF50",
            "abstained":   "#FF9800",
            "destructive": "#f44336",
            "incorrect":   "#9E9E9E",
        }

        # Build count table: agent → fault → outcome → count
        counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        for ak in agent_keys:
            diag = self._load_diagnostics(ak)
            counts[ak] = {}
            for ft in fault_types:
                trials = [t for t in diag if t["fault_type"] == ft]
                counts[ak][ft] = {oc: 0 for oc in outcome_cols}
                for t in trials:
                    oc = t.get("outcome", "incorrect")
                    if oc in counts[ak][ft]:
                        counts[ak][ft][oc] += 1
                    else:
                        counts[ak][ft]["incorrect"] += 1

        agent_labels = {
            "rule_based":  "Rule-Based",
            "aif_no_gate": "AIF (No Gate)",
            "aurora":      "AURORA",
        }
        fault_labels = {
            "network_drop": "Network Drop",
            "cpu_spike":    "CPU Spike",
            "memory_leak":  "Memory Leak",
        }

        n_agents = len(agent_keys)
        n_faults = len(fault_types)
        x        = np.arange(n_faults)
        width    = 0.25

        fig, ax = plt.subplots(figsize=(6, 3.5))

        for i, ak in enumerate(agent_keys):
            bottoms = np.zeros(n_faults)
            for oc, color in outcome_cols.items():
                vals = np.array([counts[ak][ft][oc] for ft in fault_types], dtype=float)
                bars = ax.bar(
                    x + (i - 1) * width,
                    vals,
                    width,
                    bottom=bottoms,
                    color=color,
                    label=oc.capitalize() if i == 0 else "_nolegend_",
                    edgecolor="white",
                    linewidth=0.5,
                )
                bottoms += vals

            # Agent label below each group
            for j in range(n_faults):
                ax.text(
                    x[j] + (i - 1) * width,
                    -1,
                    agent_labels[ak],
                    ha="center",
                    va="top",
                    fontsize=10,
                    rotation=45,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([fault_labels[f] for f in fault_types])
        
        ax.tick_params(axis='x', which='both', length=0)

        ax.set_ylabel("Trial Count")
        # Title removed — caption is rendered by the LaTeX figure environment
        # so the figure body keeps the same visual weight across all plots.
        ax.set_ylim(-65, 205)
        ax.set_yticks([0, 50, 100, 150])
        ax.set_yticks([], minor=True)
        # Legend on empty space outside plot, semi-transparent white background.

        ax.legend(
            title="Outcome",
            loc="upper center", ncol=4, columnspacing=1.2,
            framealpha=0.85, facecolor='white', edgecolor='0.4',
            fancybox=True, borderpad=0.35,
            handletextpad=0.4, handlelength=1.5,
            fontsize=12,
            title_fontsize=12,
        )
        ax.axhline(y=0, color="black", linewidth=0.6)

        plt.tight_layout()
        out = self.results_dir / "outcome_composition.pdf"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved outcome composition to {out}")
        plt.show()

    def plot_abstention_triggers(self):
        """
        Fig: Abstention Trigger Breakdown (AURORA only).

        Shows which safety gate caused each abstention: high_vfe,
        low_certainty, or ambiguous_ranking.  Demonstrates empirically
        that the VFE gate is the operative safety mechanism.
        """
        diag = self._load_diagnostics("aurora")
        if not diag:
            return

        reason_map = {
            "high_vfe":           r"High VFE ($\mathcal{F}$ ≥ gate)",
            "low_certainty":      r"Low Certainty ($\tau$ < gate)",
            "ambiguous_ranking":  "Ambiguous Ranking",
            "none":               "Executed (no abstention)",
        }
        reason_colors = {
            "high_vfe":           "#e65100",
            "low_certainty":      "#1565C0",
            "ambiguous_ranking":  "#6A1B9A",
            "none":               "#2e7d32",
        }

        reason_counts = {k: 0 for k in reason_map}
        for t in diag:
            r = t.get("abstain_reason", "none")
            if r not in reason_counts:
                r = "none"
            reason_counts[r] += 1

        labels  = [reason_map[k]   for k in reason_map if reason_counts[k] > 0]
        values  = [reason_counts[k] for k in reason_map if reason_counts[k] > 0]
        colors  = [reason_colors[k] for k in reason_map if reason_counts[k] > 0]

        # Compact figure: only two bars to show, so a narrow canvas with
        # wider bars uses the area better than a wide canvas with skinny bars.
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="white",
                      linewidth=0.8, width=0.7)

        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 4,
                    f"{v} ({v/len(diag)*100:.1f}%)",
                    ha="center", va="bottom", fontsize=12)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        # Y-axis label moved INSIDE the plot (upper-left) so the figure does
        # not need extra horizontal padding on the left for an outside label.
        ax.set_ylabel("Number of Trials")
        # Title removed — caption is rendered by the LaTeX figure environment.
        ax.set_ylim(0, max(values) * 1.2)

        plt.tight_layout()
        out = self.results_dir / "abstention_triggers.pdf"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved abstention triggers to {out}")
        plt.show()

    def _format_agent_name(self, name: str) -> str:
        """Format agent name for display"""
        mapping = {
            'rule_based': 'Rule-Based',
            'aif_no_gate': 'AIF (No Gate)',
            'aurora': 'AURORA (Proposed)'
        }
        return mapping.get(name, name)
    
    def _format_fault_name(self, name: str) -> str:
        """Format fault name for display"""
        mapping = {
            'network_drop': 'Network Drop',
            'cpu_spike': 'CPU Spike',
            'memory_leak': 'Memory Leak'
        }
        return mapping.get(name, name)
    
    def run_full_analysis(self):
        """
        Run complete analysis pipeline
        """
        logger.info("\n" + "="*70)
        logger.info("RUNNING COMPLETE RESULTS ANALYSIS")
        logger.info("="*70 + "\n")
        
        # 1. Comparison table
        logger.info("Generating comparison table...")
        df = self.create_comparison_table()
        print("\n" + "="*70)
        print("COMPARISON TABLE")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70 + "\n")
        
        # Save table
        table_file = self.results_dir / "comparison_table.csv"
        df.to_csv(table_file, index=False)
        logger.info(f"Saved table to {table_file}")
        self.save_confidence_intervals_json()

        # 2. Comparison chart
        logger.info("Generating comparison chart...")
        self.plot_comparison_chart()
        
        # 3. Fault type breakdown
        logger.info("Generating fault type breakdown...")
        self.plot_by_fault_type()
        
        # 4. Statistical tests
        logger.info("Running statistical significance tests...")
        self.statistical_significance_test()

        # 5. VFE distribution
        logger.info("Generating VFE distribution...")
        self.plot_vfe_distribution()

        # 6. Safety gate scatter
        logger.info("Generating safety gate scatter...")
        self.plot_safety_gate_scatter()

        # 7. Outcome composition
        logger.info("Generating outcome composition...")
        self.plot_outcome_composition()

        # 8. Abstention triggers
        logger.info("Generating abstention triggers...")
        self.plot_abstention_triggers()

        # 9. Thesis summary
        logger.info("Generating thesis summary...")
        self.generate_thesis_summary()

        logger.success("\n✓ Analysis complete! Check experiments/results/ for outputs")


# Main execution
if __name__ == "__main__":
    analyzer = ResultsAnalyzer(results_dir="experiments/results")
    analyzer.run_full_analysis()


