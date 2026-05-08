"""
Saturation analysis for the AURORA Monte-Carlo evaluation.

Answers the reviewer concern "why exactly 450 trials and not 10,000?" by
showing that the headline metrics (repair accuracy, destructive rate,
abstention rate) plateau well before N=450, and adding a power-analysis
calculation to justify the lower bound.

Two-stage design:

1. **Subsample stage (free):** the existing 450-trial run is sliced into
   prefixes {25, 50, 100, 150, 200, 300, 450} and each prefix's metrics
   (with 95% Wilson CIs) are recorded. No new trials run.

2. **Extension stage (optional):** if ``--extend`` is passed, the runner
   also produces an additional 550-trial-per-agent block so the analysis
   can extend to N=1000 and confirm the plateau holds beyond the paper's
   sample size.

Output: ``experiments/results/saturation_summary.json``
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.evaluation.proportion_ci import bootstrap_mean_ci, wilson_interval

RESULTS_DIR = ROOT / "experiments" / "results"
DEFAULT_PREFIXES = [25, 50, 100, 150, 200, 300, 450]
EXTENDED_PREFIXES = DEFAULT_PREFIXES + [600, 800, 1000, 2000, 3000, 5000, 7500, 10000]

# Per-fault subsample sizes when slicing a results JSON. Each agent has
# 150 trials per fault in the standard run, so the prefix for size N is
# the first N//3 trials of each fault.
AGENTS = [
    ("rule_based", "rule_based_results.json"),
    ("aif_no_gate", "aif_no_gate_results.json"),
    ("aurora", "aurora_results.json"),
]


def load_trials(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)["trials"]


def sample_prefix(trials: List[Dict[str, Any]], n_total: int) -> List[Dict[str, Any]]:
    """
    Take a balanced prefix of size n_total: n_total/3 trials from each fault.
    Trials are emitted in fault-major order in the original run, so this is
    just an interleaved slice.
    """
    by_fault: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in trials:
        by_fault[t["fault_type"]].append(t)
    n_per_fault = n_total // len(by_fault)
    out: List[Dict[str, Any]] = []
    for fault in sorted(by_fault):
        out.extend(by_fault[fault][:n_per_fault])
    return out


def metrics_with_ci(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute the four headline metrics on a trial slice with 95% CIs.

    - repair_accuracy: bootstrap CI on the per-trial score (continuous in [0,1])
    - destructive_rate / abstention_rate / correct_rate: Wilson CIs (binomial)
    """
    n = len(trials)
    if n == 0:
        return {"n": 0}
    n_correct = sum(1 for t in trials if t["action_outcome"] == "correct")
    n_abstain = sum(1 for t in trials if t["action_outcome"] == "abstained")
    n_destruct = sum(1 for t in trials if t["action_outcome"] == "destructive")
    scores = [float(t["repair_accuracy"]) for t in trials]

    score_lo, score_hi = bootstrap_mean_ci(scores, n_bootstrap=2000, seed=42)
    cor_lo, cor_hi = wilson_interval(n_correct, n)
    abs_lo, abs_hi = wilson_interval(n_abstain, n)
    des_lo, des_hi = wilson_interval(n_destruct, n)

    return {
        "n": n,
        "repair_accuracy": {
            "mean": sum(scores) / n,
            "ci_low": score_lo,
            "ci_high": score_hi,
        },
        "correct_rate": {
            "p_hat": n_correct / n,
            "ci_low": cor_lo,
            "ci_high": cor_hi,
            "k": n_correct,
        },
        "abstention_rate": {
            "p_hat": n_abstain / n,
            "ci_low": abs_lo,
            "ci_high": abs_hi,
            "k": n_abstain,
        },
        "destructive_rate": {
            "p_hat": n_destruct / n,
            "ci_low": des_lo,
            "ci_high": des_hi,
            "k": n_destruct,
        },
    }


def fisher_min_n(p_a: float, p_b: float, alpha: float = 0.05, power: float = 0.95) -> int:
    """
    Approximate minimum trials per arm to detect a binomial difference at the
    given power using normal approximation (good enough for the |p_a-p_b|≈0.33
    regime here, where a more exact Fisher computation would change n* by
    only a few units).
    """
    import math
    p_bar = 0.5 * (p_a + p_b)
    z_a = 1.96 if alpha == 0.05 else 2.576
    z_b = 1.645 if power == 0.95 else 1.282
    se = math.sqrt(2 * p_bar * (1 - p_bar))
    delta = abs(p_a - p_b)
    if delta < 1e-9:
        return -1
    n = ((z_a * se + z_b * math.sqrt(p_a * (1 - p_a) + p_b * (1 - p_b))) / delta) ** 2
    return int(math.ceil(n))


def extend_run(num_extra_per_fault: int, seed: int = 7) -> None:
    """
    Append an extra block of trials to each agent's results JSON, preserving
    fault balance. Useful only when ``--extend`` was requested.
    """
    from experiments.run_comparison import ExperimentRunner
    from src.aif.bayesian_network import fit_eosc_cctv_benchmark_dag
    from src.baselines.aif_no_gate_agent import AIFNoGateAgentWrapper
    from src.baselines.rule_based_agent import RuleBasedAgentWrapper
    from src.evaluation.metrics import MetricsCollector

    random.seed(seed)
    runner = ExperimentRunner(num_trials=num_extra_per_fault, output_dir=str(RESULTS_DIR))
    training_data = runner.generate_training_data(num_samples=1200)
    shared_eosc = fit_eosc_cctv_benchmark_dag(training_data)
    agents = {
        "rule_based": RuleBasedAgentWrapper("rule-baseline-ext"),
        "aif_no_gate": AIFNoGateAgentWrapper("aif-no-gate-ext"),
        "aurora": runner._create_aurora_agent(training_data, eosc_model=shared_eosc),
    }
    agents["aif_no_gate"].train(training_data, eosc_model=shared_eosc)

    for name, agent in agents.items():
        existing_path = RESULTS_DIR / f"{name}_results.json"
        if not existing_path.exists():
            logger.warning(f"Missing existing run: {existing_path}; skipping extend")
            continue
        existing = json.loads(existing_path.read_text())["trials"]
        last_id = max(t["trial_id"] for t in existing)
        collector = MetricsCollector()
        runner.trial_diagnostics = {}
        trial_id = last_id
        logger.info(f"Extending {name}: +{num_extra_per_fault * 3} trials")
        for fault in runner.fault_types:
            for _ in range(num_extra_per_fault):
                trial_id += 1
                runner.run_single_trial(trial_id, agent, name, fault, collector)
                time.sleep(0.0)
        new_trials = [t.to_dict() for t in collector.trial_metrics]
        merged = existing + new_trials
        out = {"trials": merged, "num_trials": len(merged), "timestamp": time.time()}
        existing_path.write_text(json.dumps(out, indent=2))
        logger.success(f"  → {existing_path} now has {len(merged)} trials")


def build_summary(prefixes: List[int]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"prefixes": prefixes, "agents": {}}
    for agent_key, fname in AGENTS:
        path = RESULTS_DIR / fname
        if not path.exists():
            logger.warning(f"Missing {path}; skipping agent {agent_key}")
            continue
        trials = load_trials(path)
        per_n = []
        for n in prefixes:
            if n > len(trials):
                continue
            slice_ = sample_prefix(trials, n)
            per_n.append(metrics_with_ci(slice_))
        summary["agents"][agent_key] = per_n

    summary["power_analysis"] = {
        "destructive_aif_vs_aurora": {
            "p_a": 0.333,
            "p_b": 0.000,
            "alpha": 0.05,
            "power": 0.95,
            "min_n_per_arm": fisher_min_n(0.333, 0.000),
            "note": "Sufficient n* to detect destructive-rate gap of 33pp vs 0pp at α=0.05, power=0.95.",
        },
        "repair_acc_aif_vs_aurora": {
            "p_a": 0.667,
            "p_b": 0.351,
            "alpha": 0.05,
            "power": 0.95,
            "min_n_per_arm": fisher_min_n(0.667, 0.351),
            "note": "Sufficient n* to detect 31.6pp accuracy gap.",
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extend", action="store_true",
                        help="Run additional trials so saturation extends to N=1000")
    parser.add_argument("--extra-per-fault", type=int, default=184,
                        help="Per-fault trials to append when --extend (184*3=552→ ~1000 total)")
    args = parser.parse_args()

    if args.extend:
        extend_run(args.extra_per_fault)

    # Auto-detect: if any agent already has > 450 trials, use the extended
    # prefix set so the existing data drives the saturation curve to N≈1000.
    max_trials = 0
    for _, fname in AGENTS:
        path = RESULTS_DIR / fname
        if path.exists():
            max_trials = max(max_trials, len(load_trials(path)))
    prefixes = EXTENDED_PREFIXES if max_trials > 450 else DEFAULT_PREFIXES
    summary = build_summary(prefixes)

    out = RESULTS_DIR / "saturation_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")

    print("\n=== Saturation summary (existing data) ===")
    for agent_key in ("rule_based", "aif_no_gate", "aurora"):
        rows = summary["agents"].get(agent_key, [])
        if not rows:
            continue
        print(f"\n{agent_key}:")
        print(f"  {'N':<6} {'repair_acc':<22} {'destructive':<22} {'abstain':<22}")
        for r in rows:
            ra = r["repair_accuracy"]
            dr = r["destructive_rate"]
            ab = r["abstention_rate"]
            print(
                f"  {r['n']:<6} "
                f"{ra['mean']:.3f} [{ra['ci_low']:.3f},{ra['ci_high']:.3f}]  "
                f"{dr['p_hat']:.3f} [{dr['ci_low']:.3f},{dr['ci_high']:.3f}]  "
                f"{ab['p_hat']:.3f} [{ab['ci_low']:.3f},{ab['ci_high']:.3f}]"
            )

    print("\n=== Power analysis ===")
    for k, p in summary["power_analysis"].items():
        print(f"  {k}: n* per arm ≥ {p['min_n_per_arm']} ({p['note']})")


if __name__ == "__main__":
    main()
