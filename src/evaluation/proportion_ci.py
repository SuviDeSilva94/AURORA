"""
Confidence intervals for binomial proportions (Wilson) and bootstrap CIs for means.

For thesis reporting: Wilson intervals are preferred for proportions (accuracy, abstention,
destructive rate) when you have (k, n) counts; bootstrap for continuous trial-level scores.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score interval for binomial proportion k/n.

    Returns (low, high) in [0, 1]. Edge cases: n==0 → (0, 1).
    """
    if n <= 0:
        return (0.0, 1.0)
    k = max(0, min(n, k))
    z = _z_two_sided(confidence)
    z2 = z * z
    phat = k / n
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    rad = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n) / denom
    return (max(0.0, center - rad), min(1.0, center + rad))


def _z_two_sided(confidence: float) -> float:
    # Common critical values; extend with scipy if non-standard confidence needed
    table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    if confidence in table:
        return table[confidence]
    # linear interp between 0.9 and 0.99
    if 0.9 <= confidence <= 0.99:
        t = (confidence - 0.9) / 0.09
        return 1.645 + t * (2.576 - 1.645)
    return 1.96


def wilson_summary_for_agent_metrics(metrics: Dict, confidence: float = 0.95) -> Dict[str, Dict]:
    """
    Wilson CIs for key proportions from AggregatedMetrics.to_dict() / comparison_summary.

    - repair_correct: num_correct / num_trials (strict outcome correct)
    - abstention: num_abstained / num_trials
    - destructive: num_destructive / num_trials
    """
    n = int(metrics.get("num_trials", 0))
    out: Dict[str, Dict] = {}
    specs = [
        ("repair_correct", int(metrics.get("num_correct", 0)), n),
        ("abstention", int(metrics.get("num_abstained", 0)), n),
        ("destructive", int(metrics.get("num_destructive", 0)), n),
    ]
    for name, k, nn in specs:
        lo, hi = wilson_interval(k, nn, confidence)
        p = (k / nn) if nn else 0.0
        out[name] = {
            "k": k,
            "n": nn,
            "p_hat": p,
            "ci_low": lo,
            "ci_high": hi,
            "confidence": confidence,
        }
    return out


def bootstrap_mean_ci(
    values: List[float],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of continuous trial scores."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        v = float(arr[0])
        return (v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))
