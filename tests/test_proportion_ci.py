"""Wilson / bootstrap helpers and CCTV CPT refit smoke tests."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.proportion_ci import (
    bootstrap_mean_ci,
    wilson_interval,
    wilson_summary_for_agent_metrics,
)
from src.aif.bayesian_network import (
    fit_eosc_cctv_benchmark_dag,
    max_abs_cpd_delta,
    refit_cctv_benchmark_cpts,
)


def test_wilson_interval_contains_mle():
    lo, hi = wilson_interval(45, 100)
    assert 0.0 <= lo <= 0.45 <= hi <= 1.0


def test_wilson_summary_keys():
    m = {
        "num_trials": 100,
        "num_correct": 40,
        "num_abstained": 10,
        "num_destructive": 2,
    }
    w = wilson_summary_for_agent_metrics(m)
    assert w["repair_correct"]["k"] == 40
    assert w["abstention"]["k"] == 10


def test_bootstrap_mean_ci():
    lo, hi = bootstrap_mean_ci([0.2, 0.4, 0.6, 0.8], n_bootstrap=500, seed=1)
    assert lo <= hi
    assert lo <= 0.5 <= hi


def test_refit_identical_data_zero_delta():
    rows = [
        {
            "network_quality": "high",
            "cpu": "normal",
            "memory": "normal",
            "delay": "normal",
            "slo_violated": False,
        }
    ] * 120
    m1 = fit_eosc_cctv_benchmark_dag(rows)
    m2 = refit_cctv_benchmark_cpts(rows)
    assert max_abs_cpd_delta(m1, m2) < 1e-9


def test_diagnose_markov_blanket_path_runs():
    from src.aif.root_cause_analyzer import RootCauseAnalyzer
    from src.aif import two_stage_gate as tsg

    rows = [
        {
            "network_quality": "low",
            "cpu": "high",
            "memory": "normal",
            "delay": "high",
            "slo_violated": True,
        }
    ] * 120
    m = fit_eosc_cctv_benchmark_dag(rows)
    rca = RootCauseAnalyzer(m)
    obs = dict(rows[-1])
    d = tsg.diagnose_slo_violation(
        rca, obs, top_k=3, impact_calibration=0.5, use_markov_blanket=True
    )
    assert "root_causes" in d
    s = tsg.diagnose_slo_violation_symphony(
        rca, obs, top_k=3, impact_calibration=0.5
    )
    assert "symphony_views" in s
