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

# Set plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


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
        fig.suptitle('Agent Performance Comparison', fontsize=16, fontweight='bold')
        
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
        ax1.set_title('(a) Repair correct (Wilson 95% CI)', fontweight='bold')
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
        ax2.set_title('(b) Mean Time to Repair (Lower is Better)', fontweight='bold')
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
        ax3.set_title('(c) Abstention (Wilson 95% CI)', fontweight='bold')
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
        ax4.set_title('(d) Destructive (Wilson 95% CI)', fontweight='bold')
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
        
        # Save figure
        output_file = self.results_dir / "comparison_chart.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison chart to {output_file}")
        
        plt.show()
    
    def plot_by_fault_type(self):
        """
        Plot performance breakdown by fault type
        """
        # Prepare data
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
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Performance by Fault Type', fontsize=16, fontweight='bold')
        
        # Accuracy by fault type
        pivot_acc = df.pivot(index='Fault Type', columns='Agent', values='Accuracy')
        pivot_acc.plot(kind='bar', ax=axes[0], rot=0)
        axes[0].set_ylabel('Repair Accuracy (%)', fontweight='bold')
        axes[0].set_title('(a) Repair Accuracy by Fault Type', fontweight='bold')
        axes[0].legend(title='Agent Type', loc='lower right')
        axes[0].set_ylim(0, 100)
        
        # MTTR by fault type
        pivot_mttr = df.pivot(index='Fault Type', columns='Agent', values='MTTR')
        pivot_mttr.plot(kind='bar', ax=axes[1], rot=0)
        axes[1].set_ylabel('MTTR (seconds)', fontweight='bold')
        axes[1].set_title('(b) MTTR by Fault Type', fontweight='bold')
        axes[1].legend(title='Agent Type', loc='upper right')
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.results_dir / "fault_type_breakdown.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved fault type breakdown to {output_file}")
        
        plt.show()
    
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
        
        # 5. Thesis summary
        logger.info("Generating thesis summary...")
        self.generate_thesis_summary()
        
        logger.success("\n✓ Analysis complete! Check experiments/results/ for outputs")


# Main execution
if __name__ == "__main__":
    analyzer = ResultsAnalyzer(results_dir="experiments/results")
    analyzer.run_full_analysis()


