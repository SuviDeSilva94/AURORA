"""
Two-stage safe execution: (1) diagnose root cause + certainty, (2) VFE gate on a proposed action.

Used by experiments and demos so the pipeline is explicit: diagnose → should_execute.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Maps BN root variable to healing action name (matches AURORA / coordinator).
ROOT_VARIABLE_TO_ACTION: Dict[str, str] = {
    "network_quality": "offload_to_fog",
    "cpu": "restart",
    "memory": "scale_up",
    "delay": "reduce_load",
}

# Standardized abstain labels for metrics and trial JSON.
ABSTAIN_NONE = "none"
ABSTAIN_NO_SLO = "no_slo_violation"
ABSTAIN_NO_CAUSES = "no_root_causes"
ABSTAIN_LOW_CERTAINTY = "low_certainty"
ABSTAIN_HIGH_VFE = "high_vfe"
# Top-1 vs top-2 composite ``ranking_score`` too small → structurally ambiguous (cheap gate).
ABSTAIN_AMBIGUOUS_RANKING = "ambiguous"
# Full-graph vs Markov-blanket RCA disagree on top-1 (Symphony-style dual-view consensus).
ABSTAIN_SYMPHONY = "symphony_disagree"


def ranking_margin_top1_top2(root_causes: List[Any]) -> Optional[float]:
    """
    Difference in composite RCA ranking between first and second hypotheses.

    ``None`` if fewer than two causes (no pairwise ambiguity in the returned list).
    """
    if len(root_causes) < 2:
        return None
    return float(root_causes[0].ranking_score - root_causes[1].ranking_score)


def is_ambiguous_ranking_margin(
    root_causes: List[Any], min_margin: float
) -> bool:
    """True when top-2 is within ``min_margin`` of top-1 on ``ranking_score``."""
    m = ranking_margin_top1_top2(root_causes)
    if m is None:
        return False
    return m < min_margin


def certainty_from_top_cause(top_cause: Any, impact_calibration: float) -> float:
    """
    Scalar certainty aligned with ExperimentRunner AURORA wrapper and coordinator scaling.
    """
    cal = max(impact_calibration, 1e-6)
    impact_f = min(1.0, top_cause.impact_score / cal)
    path_factor = 1.0 if len(top_cause.causal_paths) > 0 else 0.5
    return min(
        1.0,
        0.5 * impact_f + 0.3 * top_cause.evidence_strength + 0.2 * path_factor,
    )


def candidate_action_for_root_variable(variable: str) -> str:
    return ROOT_VARIABLE_TO_ACTION.get(variable, "restart")


def diagnose_slo_violation(
    rca: Any,
    obs_bn: Dict[str, Any],
    top_k: int,
    impact_calibration: float,
    *,
    use_markov_blanket: bool = False,
) -> Dict[str, Any]:
    """
    Stage 1: RCA + certainty + proposed mitigation (before VFE gate).

    ``use_markov_blanket``: restrict candidate causes to the symptom's Markov blanket
    (localized inference; ablation / second view for consensus).

    Returns:
        root_causes, top_cause, certainty, candidate_action, top_summaries (dicts for JSON).
    """
    root_causes = rca.find_root_causes(
        symptom="slo_violated",
        symptom_value=True,
        observation=obs_bn,
        top_k=top_k,
        use_markov_blanket=use_markov_blanket,
    )
    if not root_causes:
        return {
            "root_causes": [],
            "top_cause": None,
            "certainty": 0.0,
            "candidate_action": None,
            "top_summaries": [],
            "ranking_margin": None,
            "top1_ranking_score": None,
            "top2_ranking_score": None,
        }

    top = root_causes[0]
    certainty = certainty_from_top_cause(top, impact_calibration)
    candidate_action = candidate_action_for_root_variable(top.variable)
    margin = ranking_margin_top1_top2(root_causes)
    top_summaries = [
        {
            "variable": rc.variable,
            "impact": rc.impact_score,
            "evidence": rc.evidence_strength,
            "ranking_score": rc.ranking_score,
        }
        for rc in root_causes[:top_k]
    ]
    return {
        "root_causes": root_causes,
        "top_cause": top,
        "certainty": certainty,
        "candidate_action": candidate_action,
        "top_summaries": top_summaries,
        "ranking_margin": margin,
        "top1_ranking_score": float(top.ranking_score),
        "top2_ranking_score": float(root_causes[1].ranking_score)
        if len(root_causes) > 1
        else None,
    }


def diagnose_slo_violation_symphony(
    rca: Any,
    obs_bn: Dict[str, Any],
    top_k: int,
    impact_calibration: float,
) -> Dict[str, Any]:
    """
    Two RCA views (full ancestor set vs Markov-blanket restricted): abstain if top-1 differs.

    Returns the **full** diagnosis dict plus ``symphony_disagree`` and ``symphony_views``.
    When views disagree, downstream policy should abstain (``ABSTAIN_SYMPHONY``) while
    still logging ``top_cause`` from the full view for analysis.
    """
    full = diagnose_slo_violation(
        rca, obs_bn, top_k, impact_calibration, use_markov_blanket=False
    )
    mb = diagnose_slo_violation(
        rca, obs_bn, top_k, impact_calibration, use_markov_blanket=True
    )
    t_full = full["top_cause"].variable if full["top_cause"] else None
    t_mb = mb["top_cause"].variable if mb["top_cause"] else None
    full["symphony_disagree"] = False
    full["symphony_views"] = {"full_top": t_full, "markov_blanket_top": t_mb}
    if not full["root_causes"]:
        return full
    if t_full != t_mb:
        full["symphony_disagree"] = True
    return full


def should_execute_from_vfe(vfe: float, gate: Any) -> bool:
    """Stage 2: safe-execution gate (VFE below threshold)."""
    return gate.should_act(vfe)
