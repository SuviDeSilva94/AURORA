"""
Split the existing repair-accuracy score into two separate metrics:

  RCA accuracy: did the agent correctly identify the root cause? Checked
    against the ground-truth root cause derived from the injected fault
    class, regardless of whether the agent then acted or abstained.

  E2E success: did the agent both identify the right root cause AND execute
    the correct mitigation action? This is the existing num_correct counter,
    re-framed.

The split lets the thesis say: AURORA matches AIF (No Gate) on diagnosis
quality (same Bayesian Network, same posterior, same top-cause), but gives
up end-to-end success in exchange for the safety guarantee. The aggregate
repair-accuracy score conflates the two and hides this distinction.

Inputs (no re-run needed):
  experiments/results/rule_based_results.json
  experiments/results/aif_no_gate_results.json
  experiments/results/aurora_results.json

Outputs:
  experiments/results/rca_e2e_summary.json    — numerical table with Wilson CIs
  stdout                                       — formatted comparison table
"""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.proportion_ci import wilson_interval


RESULTS_DIR = Path(__file__).resolve().parent / "results"

AGENTS = {
    "Rule-Based": "rule_based_results.json",
    "AIF (No Gate)": "aif_no_gate_results.json",
    "AURORA": "aurora_results.json",
}

# Map injected fault class to the BN variable that the true root cause
# corresponds to. Used to score the agent's top cause against ground truth.
FAULT_TO_GROUND_TRUTH = {
    "network_drop": "network_quality",
    "cpu_spike": "cpu",
    "memory_leak": "memory",
}


def compute_metrics(trials: list[dict]) -> dict:
    n = len(trials)
    n_rca_correct = 0
    n_e2e_success = 0
    by_fault = {}

    for t in trials:
        fault = t["fault_type"]
        gt = FAULT_TO_GROUND_TRUTH.get(fault)
        rc = t.get("root_cause_identified")
        outcome = t.get("action_outcome")

        rca_hit = rc == gt
        e2e_hit = outcome == "correct"

        if rca_hit:
            n_rca_correct += 1
        if e2e_hit:
            n_e2e_success += 1

        bf = by_fault.setdefault(
            fault, {"n": 0, "rca": 0, "e2e": 0}
        )
        bf["n"] += 1
        bf["rca"] += int(rca_hit)
        bf["e2e"] += int(e2e_hit)

    rca_lo, rca_hi = wilson_interval(n_rca_correct, n)
    e2e_lo, e2e_hi = wilson_interval(n_e2e_success, n)

    per_fault = {}
    for fault, c in by_fault.items():
        per_fault[fault] = {
            "n": c["n"],
            "rca_accuracy_pct": 100.0 * c["rca"] / c["n"],
            "e2e_success_pct": 100.0 * c["e2e"] / c["n"],
        }

    return {
        "n_trials": n,
        "rca_accuracy_pct": 100.0 * n_rca_correct / n,
        "rca_accuracy_ci_pct": [100.0 * rca_lo, 100.0 * rca_hi],
        "e2e_success_pct": 100.0 * n_e2e_success / n,
        "e2e_success_ci_pct": [100.0 * e2e_lo, 100.0 * e2e_hi],
        "n_rca_correct": n_rca_correct,
        "n_e2e_success": n_e2e_success,
        "per_fault": per_fault,
    }


def main():
    summary = {}
    for label, fname in AGENTS.items():
        with open(RESULTS_DIR / fname) as f:
            data = json.load(f)
        summary[label] = compute_metrics(data["trials"])

    out = RESULTS_DIR / "rca_e2e_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    # Stdout table for inspection.
    print()
    print(f"{'Agent':<16s} {'RCA accuracy':<26s} {'E2E success':<26s}")
    print("-" * 70)
    for label, m in summary.items():
        rca_str = (
            f"{m['rca_accuracy_pct']:5.2f}%  "
            f"[{m['rca_accuracy_ci_pct'][0]:5.2f}, {m['rca_accuracy_ci_pct'][1]:5.2f}]"
        )
        e2e_str = (
            f"{m['e2e_success_pct']:5.2f}%  "
            f"[{m['e2e_success_ci_pct'][0]:5.2f}, {m['e2e_success_ci_pct'][1]:5.2f}]"
        )
        print(f"{label:<16s} {rca_str:<26s} {e2e_str:<26s}")
    print()
    print("Per-fault breakdown:")
    for label, m in summary.items():
        print(f"  {label}:")
        for fault, pf in m["per_fault"].items():
            print(
                f"    {fault:13s} "
                f"RCA={pf['rca_accuracy_pct']:6.2f}%  "
                f"E2E={pf['e2e_success_pct']:6.2f}%  "
                f"(n={pf['n']})"
            )
    print()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
