"""
Map continuous fault-injector observations to discrete BN states.

Must match ``generate_training_data`` in ``experiments/run_comparison.py`` so
VariableElimination / RCA receive valid CPT states (pgmpy rejects raw floats).
"""

from __future__ import annotations

from typing import Any, Dict


def discretize_observation_like_training(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of ``obs`` with ``cpu``, ``memory``, ``delay``, and optionally
    ``network_quality`` set to the same discrete labels used when learning the BN.
    """
    out = dict(obs)
    cpu = obs.get("cpu", 50.0)
    if isinstance(cpu, (int, float)):
        out["cpu"] = "high" if float(cpu) > 70 else "normal"
    memory = obs.get("memory", 50.0)
    if isinstance(memory, (int, float)):
        out["memory"] = "high" if float(memory) > 75 else "normal"
    delay = obs.get("delay", 0.0)
    if isinstance(delay, (int, float)):
        out["delay"] = "high" if float(delay) > 80 else "normal"
    net = obs.get("network")
    if isinstance(net, (int, float)):
        nf = float(net)
        out["network_quality"] = "low" if nf < 40 else ("medium" if nf < 70 else "high")
    return out
