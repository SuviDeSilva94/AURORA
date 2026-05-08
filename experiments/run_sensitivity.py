"""
One-at-a-time (OAT) sensitivity sweep over AURORA's tunable parameters.

For each parameter we hold the others at their published defaults and vary
the target parameter across a 5-point grid. At each grid point we run
``trials_per_point`` trials per fault class with one fixed BN training seed
(so the only varying input is the parameter under study).

Output:  experiments/results/sensitivity_summary.json
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from experiments.run_comparison import ExperimentRunner
from src.aif.bayesian_network import fit_eosc_cctv_benchmark_dag
from src.evaluation.metrics import MetricsCollector
from src.evaluation.proportion_ci import wilson_interval

RESULTS_DIR = ROOT / "experiments" / "results"

# Defaults match the published configuration.
DEFAULTS: Dict[str, float] = {
    "tau_base":                   0.70,
    "tau_min":                    0.50,
    "tau_lambda":                 0.15,
    "vfe_threshold":              3.85,
    "ranking_margin_threshold":   0.04,
}

# OAT sweep grids. Each list MUST contain the default value so the
# sensitivity curve passes through the published configuration.
SWEEPS: Dict[str, List[float]] = {
    "tau_base":                   [0.50, 0.60, 0.70, 0.80, 0.90],
    "tau_min":                    [0.30, 0.40, 0.50, 0.60, 0.70],
    "tau_lambda":                 [0.05, 0.10, 0.15, 0.20, 0.25],
    "vfe_threshold":              [2.50, 3.00, 3.85, 4.50, 5.00],
    "ranking_margin_threshold":   [0.00, 0.02, 0.04, 0.08, 0.12],
}

FAULT_TYPES = ["network_drop", "cpu_spike", "memory_leak"]
TRIALS_PER_POINT_PER_FAULT = 333   # 333 × 3 faults ≈ 1000 trials/point
SEED = 42


def run_one_point(
    runner: ExperimentRunner,
    eosc_model: Any,
    training_data: List[Dict[str, Any]],
    overrides: Dict[str, float],
) -> Dict[str, Any]:
    """Build an AURORA agent with the given param overrides and evaluate it."""
    # Merge overrides with defaults so unset params stay at the published value.
    cfg = {**DEFAULTS, **overrides}
    agent = runner._create_aurora_agent(
        training_data=training_data,
        eosc_model=eosc_model,
        certainty_threshold=cfg["tau_base"],
        vfe_threshold=cfg["vfe_threshold"],
        ranking_margin_threshold=cfg["ranking_margin_threshold"],
        tau_min=cfg["tau_min"],
        tau_lambda=cfg["tau_lambda"],
    )
    collector = MetricsCollector()
    runner.trial_diagnostics = {}
    trial_id = 0
    for fault in FAULT_TYPES:
        for _ in range(TRIALS_PER_POINT_PER_FAULT):
            trial_id += 1
            runner.run_single_trial(trial_id, agent, "aurora", fault, collector)

    agg = collector.get_aggregated_metrics()
    n = agg.num_trials
    des_lo, des_hi = wilson_interval(agg.num_destructive, n)
    abs_lo, abs_hi = wilson_interval(agg.num_abstained, n)
    cor_lo, cor_hi = wilson_interval(agg.num_correct, n)
    return {
        "config": dict(cfg),
        "n": n,
        "repair_accuracy": agg.mean_repair_accuracy,
        "correct_rate": agg.num_correct / n,
        "correct_ci": [cor_lo, cor_hi],
        "destructive_rate": agg.destructive_action_rate / 100.0,
        "destructive_ci": [des_lo, des_hi],
        "abstention_rate": agg.abstention_rate / 100.0,
        "abstention_ci": [abs_lo, abs_hi],
    }


def main() -> None:
    t0 = time.time()
    random.seed(SEED)
    runner = ExperimentRunner(
        num_trials=TRIALS_PER_POINT_PER_FAULT,
        output_dir=str(RESULTS_DIR),
    )
    training_data = runner.generate_training_data(num_samples=1200)
    eosc_model = fit_eosc_cctv_benchmark_dag(training_data)
    logger.info(
        f"Sensitivity sweep: {len(SWEEPS)} parameters × "
        f"{sum(len(v) for v in SWEEPS.values())} grid points × "
        f"{TRIALS_PER_POINT_PER_FAULT * len(FAULT_TYPES)} trials/point"
    )

    summary: Dict[str, Any] = {
        "defaults": DEFAULTS,
        "trials_per_point": TRIALS_PER_POINT_PER_FAULT * len(FAULT_TYPES),
        "seed": SEED,
        "sweeps": {},
    }

    for param, values in SWEEPS.items():
        rows: List[Dict[str, Any]] = []
        for v in values:
            random.seed(SEED)  # fix RNG for fault sampling so only param varies
            row = run_one_point(runner, eosc_model, training_data, {param: v})
            row["value"] = v
            rows.append(row)
            logger.info(
                f"  {param}={v:.3f} → "
                f"repair={row['repair_accuracy']:.3f} "
                f"destr={row['destructive_rate']:.3f} "
                f"abst={row['abstention_rate']:.3f}"
            )
        summary["sweeps"][param] = rows

    out = RESULTS_DIR / "sensitivity_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    # Tornado-style summary: max−min variation per metric per parameter.
    print()
    print("=== Sensitivity tornado (range of metric across the swept grid) ===")
    print(f"{'Parameter':<28} {'repair Δ':<10} {'destr Δ':<10} {'abst Δ':<10}")
    for param, rows in summary["sweeps"].items():
        rs = [r["repair_accuracy"] for r in rows]
        ds = [r["destructive_rate"] for r in rows]
        ab = [r["abstention_rate"] for r in rows]
        print(
            f"{param:<28} "
            f"{(max(rs) - min(rs))*100:>+5.1f}pp   "
            f"{(max(ds) - min(ds))*100:>+5.1f}pp   "
            f"{(max(ab) - min(ab))*100:>+5.1f}pp"
        )
    print(f"\nWrote {out}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
