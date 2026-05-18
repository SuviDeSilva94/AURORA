"""
Variational Free Energy (VFE) Computation and Safe-Execution Threshold

Only execute mitigation when VFE is below threshold (high confidence).
Root cause must first be identified with high certainty (separate check).

CCTV / benchmark BN: uses ``slo_violated`` and stress parents (``network_quality``,
``cpu``, ``memory``, ``delay``). Counterfactual queries drop ``slo_violated`` from
evidence so the model predicts compliance from (possibly action-patched) stress.
"""

from typing import Dict, Any, Optional, Union
import numpy as np
from loguru import logger

from .bayesian_network import EOSCModel

# Hypothetical post-mitigation stress states (discrete BN labels) per healing action.
# Both workloads share this dictionary because action names are unique across workloads.
HEALING_ACTION_HYPOTHESIS: Dict[str, Dict[str, str]] = {
    # CCTV workload actions
    "restart": {"cpu": "normal", "delay": "normal"},
    "scale_up": {"memory": "normal", "cpu": "normal"},
    "offload_to_fog": {"network_quality": "high", "delay": "normal"},
    "reduce_load": {"cpu": "normal", "delay": "normal"},
    # Motor-monitoring workload actions
    "vibration_alert": {"vibration": "normal", "bearing_status": "ok"},
    "thermal_throttle": {"current": "normal", "temperature": "normal"},
    "coolant_restart": {"temperature": "normal"},
    "motor_restart": {"motor_health": "healthy"},
}


def healing_hypothesis_for_action(action_name: Optional[str]) -> Dict[str, Any]:
    """
    Map a named healing action to a BN evidence patch (expected improvement on parents).

    Unknown or empty names return {} (no counterfactual change).
    """
    if not action_name:
        return {}
    return dict(HEALING_ACTION_HYPOTHESIS.get(str(action_name), {}))


class VFEComputer:
    """
    Variational Free Energy (VFE) Computer
    
    VFE = Expected Free Energy (EFE) combining:
    1. Pragmatic value: Expected reward (SLO fulfillment)
    2. Epistemic value: Information gain (uncertainty reduction)
    
    VFE = -E_q[log p(o,s|π)] + E_q[log q(s)]
        = Complexity - Accuracy
    
    Lower VFE indicates better action (less surprise, higher expected utility)
    """
    
    def __init__(self):
        self.epsilon = 1e-10  # Numerical stability

    @staticmethod
    def _is_cctv_slo_model(model: EOSCModel) -> bool:
        return "slo_violated" in model.model.nodes()

    def _filter_bn_evidence(
        self,
        model: EOSCModel,
        evidence: Dict[str, Any],
        *,
        drop_slo_violated: bool = False,
    ) -> Dict[str, Any]:
        out = {k: v for k, v in evidence.items() if k in model.model.nodes()}
        if drop_slo_violated and "slo_violated" in out:
            out = {k: v for k, v in out.items() if k != "slo_violated"}
        return out

    def _resolve_action_patch(
        self,
        model: EOSCModel,
        action: Optional[Union[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        if action is None:
            return {}
        if isinstance(action, str):
            patch = healing_hypothesis_for_action(action)
        else:
            patch = dict(action)
        return {k: v for k, v in patch.items() if k in model.model.nodes()}

    def _prob_slo_cleared(
        self,
        model: EOSCModel,
        evidence: Dict[str, Any],
        do: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        ``P(slo_violated = False | do(intervention), evidence)`` via Pearl
        do-calculus (graph mutilation performed by ``pgmpy.CausalInference``).

        ``evidence`` must not fix ``slo_violated``; any keys also present in
        ``do`` are stripped from evidence so the intervention wins. When
        ``do`` is empty, pgmpy short-circuits to a plain conditional query,
        recovering pre-do-calculus behaviour.
        """
        do_set = dict(do or {})
        ev = self._filter_bn_evidence(model, evidence, drop_slo_violated=True)
        ev = {k: v for k, v in ev.items() if k not in do_set}
        try:
            qr = model.causal_inference.query(
                variables=["slo_violated"],
                do=do_set,
                evidence=ev,
                adjustment_set=[],
                show_progress=False,
            )
            vals = np.asarray(qr.values).flatten()
            cpd = model.model.get_cpds("slo_violated")
            states = list(cpd.state_names["slo_violated"])
            for i, s in enumerate(states):
                if s is False:
                    return float(vals[i])
            return float(vals[0]) if len(vals) >= 1 else 0.5
        except Exception as e:
            logger.debug(f"slo_violated clear probability failed: {e}")
            return 0.5

    def compute_vfe_decomposed(
        self,
        model: EOSCModel,
        evidence: Dict[str, Any],
        action: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """
        VFE with explicit pragmatic vs epistemic terms (CCTV BN).

        - pragmatic: ``-log P(SLO clear | do(action), evidence)`` — Pearl
          interventional probability via graph mutilation; prefer low.
        - epistemic: mean KL between predictive marginals before/after the
          action — a belief-shift metric over the model's posterior; prefer low.
        - vfe: epistemic + pragmatic (lower is better).
        """
        patch = self._resolve_action_patch(model, action)

        if self._is_cctv_slo_model(model):
            obs_evidence = self._filter_bn_evidence(
                model, evidence, drop_slo_violated=True
            )
            # Pragmatic term — interventional via do-calculus.
            p_clear = self._prob_slo_cleared(model, obs_evidence, do=patch)
            pragmatic = -float(np.log(p_clear + self.epsilon))
            # Epistemic term — KL between predictive marginals before vs after
            # the action. Build the post-action evidence by overlaying the
            # patch (the patched node is fixed at its expected post-action
            # state for downstream-marginal prediction).
            post = {
                **{k: v for k, v in obs_evidence.items() if k not in patch},
                **patch,
            }
            epistemic = self._compute_complexity(model, post, obs_evidence)
            vfe = float(epistemic + pragmatic)
            return {"vfe": vfe, "pragmatic": pragmatic, "epistemic": epistemic}

        # Legacy head: reuse averaged log SLO fulfillment as "accuracy"
        hypo = evidence.copy()
        hypo.update(patch)
        accuracy = self._compute_accuracy_legacy(model, hypo)
        epistemic = self._compute_complexity(model, hypo, evidence)
        vfe = float(epistemic - accuracy)
        pragmatic = float(-accuracy)
        return {"vfe": vfe, "pragmatic": pragmatic, "epistemic": epistemic}

    def compute_vfe(
        self,
        model: EOSCModel,
        evidence: Dict[str, Any],
        action: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> float:
        """
        Compute Variational Free Energy for a proposed action.

        ``action`` may be a patch dict, a healing action name (``restart``, ``scale_up``,
        ``offload_to_fog``, ``reduce_load``), or empty for no counterfactual change.
        """
        return float(self.compute_vfe_decomposed(model, evidence, action)["vfe"])
    
    def _compute_accuracy_legacy(
        self,
        model: EOSCModel,
        evidence: Dict[str, Any]
    ) -> float:
        """
        Legacy accuracy: average log P(SLO "high" / fulfilled) for old variable names.
        """
        slo_variables = ['network', 'in_time', 'success', 'distance']
        
        total_log_prob = 0.0
        count = 0
        
        for slo in slo_variables:
            if slo in model.model.nodes():
                prob = model.query_slo(slo, evidence)
                log_prob = np.log(prob + self.epsilon)
                total_log_prob += log_prob
                count += 1
        
        return total_log_prob / max(count, 1)

    def _compute_complexity(
        self,
        model: EOSCModel,
        new_evidence: Dict[str, Any],
        old_evidence: Dict[str, Any]
    ) -> float:
        """
        Epistemic term: mean KL between marginals after vs before the action patch.
        """
        # Get predictions under both evidence sets
        try:
            old_predictions = model.predict(old_evidence)
            new_predictions = model.predict(new_evidence)
            
            # Compute KL divergence between distributions
            kl_div = 0.0
            count = 0
            
            for var in old_predictions:
                if var in new_predictions:
                    old_dist = old_predictions[var]
                    new_dist = new_predictions[var]
                    
                    # KL(new || old)
                    kl = self._kl_divergence(new_dist, old_dist)
                    kl_div += kl
                    count += 1
            
            complexity = kl_div / max(count, 1)
            
        except Exception as e:
            logger.debug(f"Error computing complexity: {e}")
            complexity = 0.0
        
        return complexity
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute KL divergence: KL(P || Q) = Σ p(x) log(p(x)/q(x))
        
        Args:
            p: Probability distribution P
            q: Probability distribution Q
            
        Returns:
            KL divergence
        """
        p = np.array(p) + self.epsilon
        q = np.array(q) + self.epsilon
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        kl = np.sum(p * np.log(p / q))
        return float(kl)
    
    def compute_expected_free_energy(
        self,
        model: EOSCModel,
        evidence: Dict[str, Any],
        action: Dict[str, Any],
        preferences: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute Expected Free Energy (EFE) with explicit preferences
        
        EFE = Pragmatic Value + Epistemic Value
            = -E[log P(o|π)] - E[H[s|o,π]]
        
        Args:
            model: EOSC model
            evidence: Current observations
            action: Proposed action
            preferences: Preferred states (SLO targets)
            
        Returns:
            EFE value
        """
        # Pragmatic value: Expected reward (SLO fulfillment)
        pragmatic_value = self._compute_pragmatic_value(
            model, evidence, action, preferences
        )
        
        # Epistemic value: Information gain
        epistemic_value = self._compute_epistemic_value(
            model, evidence, action
        )
        
        # EFE (lower is better)
        efe = -pragmatic_value - epistemic_value
        
        return efe
    
    def _compute_pragmatic_value(
        self,
        model: EOSCModel,
        evidence: Dict[str, Any],
        action: Dict[str, Any],
        preferences: Optional[Dict[str, float]]
    ) -> float:
        """
        Compute pragmatic value: expected utility ``P(SLO clear | do(action))``.

        For the CCTV BN this routes through Pearl do-calculus (graph
        mutilation) rather than observational conditioning.
        """
        if self._is_cctv_slo_model(model):
            patch = {k: v for k, v in (action or {}).items() if k in model.model.nodes()}
            obs_evidence = self._filter_bn_evidence(
                model, evidence, drop_slo_violated=True
            )
            return self._prob_slo_cleared(model, obs_evidence, do=patch)

        hypothetical_evidence = evidence.copy()
        hypothetical_evidence.update(action)
        slo_variables = ['network', 'in_time', 'success', 'distance']
        total_value = 0.0
        n = 0
        for slo in slo_variables:
            if slo in model.model.nodes():
                prob = model.query_slo(slo, hypothetical_evidence)
                weight = preferences.get(slo, 1.0) if preferences else 1.0
                total_value += prob * weight
                n += 1
        return total_value / max(n, 1)
    
    def _compute_epistemic_value(
        self,
        model: EOSCModel,
        evidence: Dict[str, Any],
        action: Dict[str, Any]
    ) -> float:
        """
        Compute epistemic value: Expected information gain
        
        Measures uncertainty reduction from taking action
        """
        hypothetical_evidence = evidence.copy()
        hypothetical_evidence.update(action)
        
        # Compute entropy before and after action
        entropy_before = self._compute_entropy(model, evidence)
        entropy_after = self._compute_entropy(model, hypothetical_evidence)
        
        # Information gain
        info_gain = entropy_before - entropy_after
        
        return info_gain
    
    def _compute_entropy(
        self,
        model: EOSCModel,
        evidence: Dict[str, Any]
    ) -> float:
        """
        Compute entropy of belief distribution
        
        H[s] = -Σ p(s) log p(s)
        """
        predictions = model.predict(evidence)
        
        total_entropy = 0.0
        count = 0
        
        for var, dist in predictions.items():
            entropy = -np.sum(dist * np.log(dist + self.epsilon))
            total_entropy += entropy
            count += 1
        
        return total_entropy / max(count, 1)


class UncertaintyGate:
    """
    Safe-Execution Gate: only act when VFE is below threshold.
    
    After root cause is identified with high certainty, this gate checks
    that the agent is confident enough to execute (low VFE). When VFE
    exceeds threshold, the agent abstains from action and continues
    monitoring and learning.
    """
    
    def __init__(self, threshold: float = 8.0):
        """
        Initialize safe-execution gate (VFE threshold)
        
        Args:
            threshold: VFE threshold above which to abstain from action
                      Higher threshold = more risk tolerance
                      Lower threshold = more conservative
        """
        self.threshold = threshold
        self.abstention_history: list = []
    
    def should_act(self, vfe: float) -> bool:
        """
        Determine if agent should execute action based on VFE
        
        Args:
            vfe: Computed Variational Free Energy
            
        Returns:
            True if VFE < threshold (safe to act)
            False if VFE >= threshold (abstain)
        """
        should_act = vfe < self.threshold
        
        self.abstention_history.append({
            'vfe': vfe,
            'acted': should_act
        })
        
        if not should_act:
            logger.warning(
                f"Safe-execution gate: abstaining (VFE={vfe:.3f} >= {self.threshold:.3f})"
            )
        
        return should_act
    
    def get_abstention_rate(self) -> float:
        """Get percentage of times agent abstained"""
        if not self.abstention_history:
            return 0.0
        
        abstentions = sum(1 for h in self.abstention_history if not h['acted'])
        return abstentions / len(self.abstention_history)
    
    def adapt_threshold(self, success_rate: float):
        """
        Adaptively adjust threshold based on success rate
        
        If agent is consistently successful, can afford higher risk (higher threshold)
        If agent is failing, be more conservative (lower threshold)
        
        Args:
            success_rate: SLO fulfillment rate
        """
        if success_rate > 0.9:
            # Increase threshold (more risk tolerance)
            self.threshold = min(5.0, self.threshold * 1.1)
        elif success_rate < 0.5:
            # Decrease threshold (more conservative)
            self.threshold = max(0.5, self.threshold * 0.9)
        
        logger.info(f"Adapted VFE threshold to {self.threshold:.3f} (SLO rate={success_rate:.2%})")


