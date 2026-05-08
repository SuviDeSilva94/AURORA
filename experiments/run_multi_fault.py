"""
Multi-fault experiment driver — exercises Eq. (11) by injecting concurrent
SLO violations so |At| varies from 1 to 3 across trials.

Output: experiments/results/aurora_multi_fault_diagnostics.json
        experiments/results/aurora_multi_fault_results.json

Used to answer the reviewer concern that Gate 1 fired 0/450 times in the
single-fault sweep. With compound faults, posterior concentration drops as
|At| grows (paper §III-C-2) and Gate 1 — under the dynamic τ schedule —
becomes load-bearing.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from experiments.run_comparison import ExperimentRunner, FaultInjector
from src.aif.bayesian_network import fit_eosc_cctv_benchmark_dag
from src.evaluation.metrics import ActionOutcome, MetricsCollector


# Compound fault profiles. Naming convention: "compound_<a>+<b>".
# The dominant fault (used for ground-truth labelling) is the FIRST listed,
# which is the one whose patch in Table I is most likely to dominate the
# overlay. Compound trials primarily exist to drive |At| upwards; per-trial
# repair-accuracy is informative but secondary.
COMPOUND_PROFILES: Dict[str, List[str]] = {
    "compound_net+cpu":      ["network_drop", "cpu_spike"],
    "compound_net+mem":      ["network_drop", "memory_leak"],
    "compound_cpu+mem":      ["cpu_spike",    "memory_leak"],
    "compound_net+cpu+mem":  ["network_drop", "cpu_spike", "memory_leak"],
}


def _make_compound_injector(parts: List[str]):
    def _inj(severity: float = 0.7) -> Dict[str, Any]:
        return FaultInjector.inject_compound_fault(parts, severity=severity)
    return _inj


def build_runner(num_trials_per_profile: int) -> ExperimentRunner:
    """
    Configure the standard runner with compound fault profiles registered.
    """
    runner = ExperimentRunner(
        num_trials=num_trials_per_profile,
        output_dir="experiments/results",
    )

    for name, parts in COMPOUND_PROFILES.items():
        runner.fault_types[name] = _make_compound_injector(parts)
        # Ground-truth labels follow the dominant (first) fault. This is a
        # convention, not a claim that compound faults have a single root
        # cause; the gate-firing analysis does not depend on it.
        dominant = parts[0]
        runner.ground_truth_root_causes[name] = (
            runner.ground_truth_root_causes[dominant]
        )
        runner.correct_actions[name] = runner.correct_actions[dominant]

    return runner


def main(num_trials_per_profile: int = 400, seed: int = 42) -> None:
    random.seed(seed)
    runner = build_runner(num_trials_per_profile)

    training_data = runner.generate_training_data(num_samples=1200)
    shared_eosc = fit_eosc_cctv_benchmark_dag(training_data)
    aurora = runner._create_aurora_agent(
        training_data,
        eosc_model=shared_eosc,
    )

    collector = MetricsCollector()
    runner.trial_diagnostics = {}
    trial_counter = 0

    profiles = list(runner.fault_types.keys())
    logger.info(
        f"Multi-fault sweep: {len(profiles)} profiles × "
        f"{num_trials_per_profile} trials = "
        f"{len(profiles) * num_trials_per_profile} trials"
    )

    for fault_type in profiles:
        for _ in range(num_trials_per_profile):
            trial_counter += 1
            runner.run_single_trial(
                trial_id=trial_counter,
                agent_wrapper=aurora,
                agent_type="aurora",
                fault_type=fault_type,
                collector=collector,
            )
            time.sleep(0.0)

    out_diag = runner.output_dir / "aurora_multi_fault_diagnostics.json"
    out_res = runner.output_dir / "aurora_multi_fault_results.json"
    with open(out_diag, "w") as f:
        json.dump(runner.trial_diagnostics.get("aurora", []), f, indent=2)
    collector.save_results(str(out_res))

    diags = runner.trial_diagnostics.get("aurora", [])
    by_n: Dict[int, int] = {}
    g1: Dict[int, int] = {}
    g2: Dict[int, int] = {}
    for d in diags:
        n = d.get("anomaly_count")
        if n is None:
            continue
        by_n[n] = by_n.get(n, 0) + 1
        if d.get("gate1_fired"):
            g1[n] = g1.get(n, 0) + 1
        if d.get("gate2_fired"):
            g2[n] = g2.get(n, 0) + 1

    print("\n=== Multi-fault gate firing summary ===")
    print(f"{'|At|':<6} {'trials':<8} {'Gate1 fired':<14} {'Gate2 fired':<14}")
    for n in sorted(by_n):
        total = by_n[n]
        print(
            f"{n:<6} {total:<8} "
            f"{g1.get(n, 0):<5} ({100*g1.get(n,0)/total:.1f}%)  "
            f"{g2.get(n, 0):<5} ({100*g2.get(n,0)/total:.1f}%)"
        )
    print(f"\nDiagnostics: {out_diag}")
    print(f"Results:     {out_res}")


if __name__ == "__main__":
    main()
