"""
Multi-seed sensitivity analysis: characterize how destructive rate, repair
accuracy, and abstention vary across BN training seeds.

Motivation: extending the original experiment to N=1000 revealed that AURORA's
"0% destructive" headline (Table II) is not stable — under a re-trained BN it
drifts to ~5.7%. This script quantifies the variance:

  for seed in {1..5}:
      training_data ← simulator(seed)
      BN ← fit_eosc_cctv_benchmark_dag(training_data)
      for agent in [AURORA, AIF-no-gate]:
          run 200 trials on this BN
          record (repair_acc, destructive_rate, abstention_rate)

Aggregate across seeds → mean ± SD per metric.
Output: experiments/results/multi_seed_summary.json
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from experiments.run_comparison import ExperimentRunner
from src.aif.bayesian_network import fit_eosc_cctv_benchmark_dag
from src.baselines.aif_no_gate_agent import AIFNoGateAgentWrapper
from src.evaluation.metrics import MetricsCollector
from src.evaluation.proportion_ci import wilson_interval

RESULTS_DIR = ROOT / "experiments" / "results"

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
TRIALS_PER_SEED = 1000         # 334/333/333 across the three faults; 10K per agent
FAULT_TYPES = ["network_drop", "cpu_spike", "memory_leak"]


def trials_per_fault(total: int, n_faults: int) -> List[int]:
    base = total // n_faults
    rem = total - base * n_faults
    return [base + (1 if i < rem else 0) for i in range(n_faults)]


def run_one_seed(seed: int) -> Dict[str, Any]:
    """Train one BN, evaluate AURORA + AIF-no-gate on it, return metrics."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    runner = ExperimentRunner(
        num_trials=TRIALS_PER_SEED,
        output_dir=str(RESULTS_DIR),
    )
    training_data = runner.generate_training_data(num_samples=1200)
    shared_eosc = fit_eosc_cctv_benchmark_dag(training_data)

    aurora = runner._create_aurora_agent(training_data, eosc_model=shared_eosc)
    aif = AIFNoGateAgentWrapper(f"aif-no-gate-seed{seed}")
    aif.train(training_data, eosc_model=shared_eosc)

    out: Dict[str, Any] = {"seed": seed, "agents": {}}
    splits = trials_per_fault(TRIALS_PER_SEED, len(FAULT_TYPES))

    for agent_key, agent in [("aurora", aurora), ("aif_no_gate", aif)]:
        collector = MetricsCollector()
        runner.trial_diagnostics = {}
        trial_id = 0
        for fault, k in zip(FAULT_TYPES, splits):
            for _ in range(k):
                trial_id += 1
                runner.run_single_trial(
                    trial_id, agent, agent_key, fault, collector
                )
        agg = collector.get_aggregated_metrics()
        n = agg.num_trials
        des_lo, des_hi = wilson_interval(agg.num_destructive, n)
        cor_lo, cor_hi = wilson_interval(agg.num_correct, n)
        abs_lo, abs_hi = wilson_interval(agg.num_abstained, n)
        out["agents"][agent_key] = {
            "n": n,
            "repair_accuracy": agg.mean_repair_accuracy,
            "destructive_rate": agg.destructive_action_rate / 100.0,
            "destructive_count": agg.num_destructive,
            "destructive_ci": [des_lo, des_hi],
            "abstention_rate": agg.abstention_rate / 100.0,
            "abstention_count": agg.num_abstained,
            "abstention_ci": [abs_lo, abs_hi],
            "correct_count": agg.num_correct,
            "correct_ci": [cor_lo, cor_hi],
            # Failure-mode breakdown: count cpu_spike→scale_up specifically
            # (the failure mode the saturation extension exposed).
            "destructive_by_fault": _destr_by_fault(collector),
        }
    return out


def _destr_by_fault(collector: MetricsCollector) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in collector.trial_metrics:
        if t.action_outcome == "destructive":
            out[t.fault_type] = out.get(t.fault_type, 0) + 1
    return out


def aggregate(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean ± SD across seeds, per agent per metric."""
    agg: Dict[str, Any] = {"per_seed": per_seed, "summary": {}}
    for agent_key in ("aurora", "aif_no_gate"):
        rows = [r["agents"][agent_key] for r in per_seed]
        des = [r["destructive_rate"] for r in rows]
        repair = [r["repair_accuracy"] for r in rows]
        abst = [r["abstention_rate"] for r in rows]
        agg["summary"][agent_key] = {
            "destructive_rate": {
                "mean": mean(des),
                "std": stdev(des) if len(des) > 1 else 0.0,
                "min": min(des),
                "max": max(des),
                "values": des,
            },
            "repair_accuracy": {
                "mean": mean(repair),
                "std": stdev(repair) if len(repair) > 1 else 0.0,
                "min": min(repair),
                "max": max(repair),
                "values": repair,
            },
            "abstention_rate": {
                "mean": mean(abst),
                "std": stdev(abst) if len(abst) > 1 else 0.0,
                "min": min(abst),
                "max": max(abst),
                "values": abst,
            },
        }
    return agg


def main() -> None:
    t0 = time.time()
    per_seed: List[Dict[str, Any]] = []
    for seed in SEEDS:
        logger.info(f"=== Seed {seed} ===")
        per_seed.append(run_one_seed(seed))

    summary = aggregate(per_seed)
    out_path = RESULTS_DIR / "multi_seed_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print(f"=== Multi-seed summary ({len(SEEDS)} seeds × {TRIALS_PER_SEED} trials) ===")
    for agent_key in ("aurora", "aif_no_gate"):
        s = summary["summary"][agent_key]
        print(f"\n{agent_key}:")
        for metric in ("repair_accuracy", "destructive_rate", "abstention_rate"):
            m = s[metric]
            vs = ", ".join(f"{v:.3f}" for v in m["values"])
            print(f"  {metric:<18} mean={m['mean']:.3f} ± {m['std']:.3f}  "
                  f"range=[{m['min']:.3f}, {m['max']:.3f}]  per-seed=[{vs}]")

    print("\nPer-seed destructive breakdown by fault (AURORA):")
    for r in per_seed:
        sd = r["agents"]["aurora"]["destructive_by_fault"]
        print(f"  seed={r['seed']}: {sd or '{}'}")

    print(f"\nWrote {out_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
