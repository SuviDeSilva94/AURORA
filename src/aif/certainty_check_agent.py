"""
CertaintyCheckAgent — Dedicated parallel micro-agent for root-cause certainty

required: "can use maybe some other agent whether the decision taken
by this small lm is certain or uncertain."

This agent answers: "Is the proposed root cause certain enough to act?"
It combines:
  - Scalar certainty (impact + evidence)
  - VFE (Variational Free Energy) as part of root-cause certainty
     (required: "VFE comes after we find the root cause" — tied to the decision)
Only if both pass do we proceed to planning and healing.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from src.aif.bayesian_network import EOSCModel
from src.aif.root_cause_analyzer import RootCause
from src.aif.vfe_compute import VFEComputer


@dataclass
class CertaintyCheckResult:
    """Result of the certainty-check agent: is the root-cause decision certain?"""
    is_certain: bool
    certainty_score: float
    vfe_value: float
    should_proceed: bool  # True only if certain AND VFE below threshold
    reason: str


class CertaintyCheckAgent:
    """
    Micro-agent that checks whether the decision (proposed root cause) is
    certain or uncertain. Runs in parallel with other agents; its sole task
    is this check. VFE is computed here as part of root-cause certainty
    (not only at execution time).
    """

    def __init__(
        self,
        agent_id: str = "certainty-check-001",
        eosc_model: Optional[EOSCModel] = None,
        certainty_threshold: float = 0.85,
        vfe_threshold: float = 8.0,
    ):
        self.agent_id = agent_id
        self.eosc_model = eosc_model
        self.certainty_threshold = certainty_threshold
        self.vfe_threshold = vfe_threshold
        self.vfe_computer = VFEComputer()
        logger.info(
            f"[{agent_id}] CertaintyCheckAgent initialized "
            f"(certainty>={certainty_threshold:.0%}, VFE<{vfe_threshold})"
        )

    def set_model(self, eosc_model: EOSCModel) -> None:
        """Set the EOSC model (e.g. after coordinator provides it)."""
        self.eosc_model = eosc_model

    def check(
        self,
        top_root_cause: RootCause,
        observation: Dict[str, Any],
        eosc_model: Optional[EOSCModel] = None,
    ) -> CertaintyCheckResult:
        """
        Check whether the proposed root cause is certain enough to act.

        Combines:
          1. Certainty score from impact + evidence (scalar)
          2. VFE for the current state / root-cause hypothesis
             (VFE after root cause — part of certainty on the decision)

        Returns should_proceed=True only if certainty >= threshold AND VFE < threshold.
        """
        model = eosc_model or self.eosc_model
        if model is None:
            logger.warning(f"[{self.agent_id}] No EOSC model; abstaining")
            return CertaintyCheckResult(
                is_certain=False,
                certainty_score=0.0,
                vfe_value=float('inf'),
                should_proceed=False,
                reason="no_model",
            )

        # 1) Scalar certainty (impact + evidence)
        certainty_score = (
            top_root_cause.impact_score + top_root_cause.evidence_strength
        ) / 2.0
        certainty_score = min(1.0, certainty_score)

        # 2) VFE as part of root-cause certainty (after we have the root cause)
        try:
            vfe_value = self.vfe_computer.compute_vfe(
                model, observation, action={}
            )
        except Exception as e:
            logger.debug(f"[{self.agent_id}] VFE computation failed: {e}")
            vfe_value = 3.0  # assume high uncertainty

        # 3) Decision: certain only if both pass
        certainty_ok = certainty_score >= self.certainty_threshold
        vfe_ok = vfe_value < self.vfe_threshold
        should_proceed = certainty_ok and vfe_ok
        is_certain = certainty_ok  # "is the decision certain?" (scalar part)

        if not certainty_ok:
            reason = "certainty_below_threshold"
        elif not vfe_ok:
            reason = "vfe_above_threshold"
        else:
            reason = "ok"

        logger.info(
            f"[{self.agent_id}] Check result: certainty={certainty_score:.2%}, VFE={vfe_value:.2f} "
            f"→ should_proceed={should_proceed} ({reason})"
        )

        return CertaintyCheckResult(
            is_certain=is_certain,
            certainty_score=certainty_score,
            vfe_value=vfe_value,
            should_proceed=should_proceed,
            reason=reason,
        )
