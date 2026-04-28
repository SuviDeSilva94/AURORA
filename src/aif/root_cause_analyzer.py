"""
Root Cause Analysis using Active Inference and Bayesian Networks

This module implements causal discovery and root cause identification
for distributed micro-agents in the Computing Continuum.

The goal is NOT to just fix problems, but to UNDERSTAND WHY they happened.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from src.aif.bayesian_network import EOSCModel


@dataclass
class CausalPath:
    """
    Represents a causal chain from root cause to symptom
    
    Example:
        faulty_load_balancer → too_many_connections → 
        network_congestion → high_delay → slo_violated
    """
    path: List[str]  # Sequence of variables
    strength: float  # How strong is this causal relationship?
    confidence: float  # How confident are we?
    explanation: str  # Human-readable explanation
    
    def __str__(self):
        return " → ".join(self.path) + f" (strength={self.strength:.2f})"


@dataclass
class RootCause:
    """
    A root cause with evidence and impact assessment
    """
    variable: str  # The root cause variable
    value: Any  # The problematic value
    impact_score: float  # How much does this affect SLOs?
    causal_paths: List[CausalPath]  # All paths from this cause to symptoms
    evidence_strength: float  # How strong is the evidence?
    explanation: str  # Why is this a root cause?
    ranking_score: float = 0.0  # Composite rank score (impact + evidence + topology)
    
    def __str__(self):
        return f"Root Cause: {self.variable}={self.value} (impact={self.impact_score:.2f})"


class RootCauseAnalyzer:
    """
    Analyzes Bayesian Networks to discover root causes of SLO violations.
    
    This uses:
    1. Causal graph structure (BN topology)
    2. Conditional probabilities (CPTs)
    3. Counterfactual reasoning ("What if X was different?")
    4. Information flow analysis
    
    Based on:
    - Pearl, J. (2009). Causality: Models, Reasoning, and Inference
    - Sedlak et al. (2024) - Active Inference for Computing Continuum
    """
    
    def __init__(self, eosc_model: EOSCModel):
        self.model = eosc_model
        self.causal_graph = eosc_model.model
    
    def find_root_causes(
        self, 
        symptom: str, 
        symptom_value: Any,
        observation: Dict[str, Any],
        top_k: int = 5,
        use_markov_blanket: bool = False,
    ) -> List[RootCause]:
        """
        Find root causes of a symptom (e.g., SLO violation).
        Optionally restrict to Markov blanket of the symptom (Sedlak 2024: localized inference).
        
        Args:
            symptom: The observed problem (e.g., 'slo_violated')
            symptom_value: The problematic value (e.g., True)
            observation: Current system state
            top_k: Return top K root causes
            use_markov_blanket: If True, consider only causes in the symptom's Markov blanket
        """
        logger.info(f"Finding root causes for {symptom}={symptom_value}")
        
        # ═══════════════════════════════════════════════════════════════
        # Step 1: Find potential causes (ancestors, optionally within Markov blanket)
        # ═══════════════════════════════════════════════════════════════
        ancestors = self._find_ancestors(symptom)
        if use_markov_blanket and hasattr(self.model, 'get_markov_blanket'):
            mb = self.model.get_markov_blanket(symptom)
            ancestors = ancestors & mb  # Restrict to MB for localized backtracking (Sedlak 2024)
            logger.debug(f"Restricted to Markov blanket of {symptom}: {ancestors}")
        logger.debug(f"Found {len(ancestors)} potential causes: {ancestors}")
        
        # ═══════════════════════════════════════════════════════════════
        # Step 2: For each potential cause, compute causal impact
        # ═══════════════════════════════════════════════════════════════
        root_causes = []
        
        for cause_var in ancestors:
            if cause_var not in observation:
                continue
            
            # Compute how much this variable contributes to the symptom
            impact = self._compute_causal_impact(
                cause=cause_var,
                symptom=symptom,
                observation=observation
            )
            
            # Compute evidence strength
            evidence = self._compute_evidence_strength(
                cause_var, 
                observation
            )

            # Filter weak causes, but keep high-evidence stress signals
            if impact < 0.1 and evidence < 0.9:
                continue

            # Find all causal paths from cause to symptom
            paths = self._find_causal_paths(cause_var, symptom)
            
            # Generate explanation
            explanation = self._generate_explanation(
                cause_var,
                symptom,
                paths,
                observation
            )
            
            ranking_score = self._compute_ranking_score(
                cause_var=cause_var,
                symptom=symptom,
                impact=impact,
                evidence=evidence,
            )

            root_cause = RootCause(
                variable=cause_var,
                value=observation[cause_var],
                impact_score=impact,
                causal_paths=paths,
                evidence_strength=evidence,
                ranking_score=ranking_score,
                explanation=explanation
            )
            
            root_causes.append(root_cause)
        
        # ═══════════════════════════════════════════════════════════════
        # Step 3: Rank by composite score (impact + evidence + direct-parent bonus)
        # ═══════════════════════════════════════════════════════════════
        root_causes.sort(
            key=lambda rc: (rc.ranking_score, rc.impact_score, rc.evidence_strength),
            reverse=True,
        )
        
        return root_causes[:top_k]

    def _compute_ranking_score(
        self,
        cause_var: str,
        symptom: str,
        impact: float,
        evidence: float,
    ) -> float:
        """
        Composite root-cause ranking score.

        We prioritize:
        1) Counterfactual impact (primary signal),
        2) Observation evidence strength,
        3) Topology hint: direct parent of symptom.
        """
        is_direct_parent = 1.0 if cause_var in set(self.causal_graph.predecessors(symptom)) else 0.0
        has_parent = len(list(self.causal_graph.predecessors(cause_var))) > 0
        root_bonus = 0.15 if not has_parent else 0.0
        mediator_penalty = 0.10 if has_parent else 0.0
        return (
            0.65 * impact
            + 0.25 * evidence
            + 0.10 * is_direct_parent
            + root_bonus
            - mediator_penalty
        )
    
    def _find_ancestors(self, node: str) -> Set[str]:
        """
        Find all ancestors of a node in the causal graph.
        
        Ancestors are variables that can causally influence this node.
        These are potential root causes.
        """
        if node not in self.causal_graph.nodes():
            return set()
        
        ancestors = set()
        to_visit = [node]
        visited = set()
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)
            
            # Get parents (direct causes)
            parents = list(self.causal_graph.predecessors(current))
            
            for parent in parents:
                ancestors.add(parent)
                to_visit.append(parent)
        
        return ancestors
    
    def _compute_causal_impact(
        self, 
        cause: str, 
        symptom: str,
        observation: Dict[str, Any]
    ) -> float:
        """
        Compute how much the cause variable impacts the symptom.
        
        This uses counterfactual reasoning:
        "If the cause had a different value, how would the symptom change?"
        
        Impact = P(symptom=bad | cause=observed) - P(symptom=bad | cause=good)
        """
        try:
            import networkx as nx
            descendants = nx.descendants(self.causal_graph, cause)
            
            # Current state: P(symptom | cause=observed). 
            # To measure true interventional effect do(cause), we must NOT condition on its descendants (mediators).
            evidence_current = {k: v for k, v in observation.items() 
                               if k in self.causal_graph.nodes() and k != symptom and k not in descendants}
            
            result_current = self.model.inference_engine.query(
                variables=[symptom],
                evidence=evidence_current
            )
            prob_current = result_current.values[1] if len(result_current.values) > 1 else result_current.values[0]
            
            # Counterfactual: flip cause to a "better" discrete state (matches BN CPTs)
            evidence_counterfactual = evidence_current.copy()
            current_value = observation[cause]
            if isinstance(current_value, (int, float)):
                counterfactual_value = max(0, float(current_value) * 0.5)
                evidence_counterfactual[cause] = int(counterfactual_value)
            elif isinstance(current_value, str):
                v = current_value.lower()
                # CPT states differ per variable; never use invalid labels (e.g. "normal" for network_quality).
                benign_by_cause = {
                    "network_quality": "high",
                    "cpu": "normal",
                    "memory": "normal",
                    "delay": "normal",
                }
                worst_by_cause = {
                    "network_quality": "low",
                    "cpu": "high",
                    "memory": "high",
                    "delay": "high",
                }
                if cause in benign_by_cause:
                    benign = benign_by_cause[cause]
                    worst = worst_by_cause[cause]
                    evidence_counterfactual[cause] = worst if v == benign else benign
                else:
                    flips = {
                        "high": "normal",
                        "normal": "high",
                        "low": "high",
                        "medium": "high",
                    }
                    alt = flips.get(v)
                    if alt is None:
                        return 0.5
                    evidence_counterfactual[cause] = alt
            else:
                return 0.5
            
            result_counter = self.model.inference_engine.query(
                variables=[symptom],
                evidence=evidence_counterfactual
            )
            prob_counter = result_counter.values[1] if len(result_counter.values) > 1 else result_counter.values[0]
            
            # Impact = How much the symptom probability changed
            impact = abs(prob_current - prob_counter)
            
            logger.debug(f"Causal impact {cause}→{symptom}: {impact:.3f}")
            
            return impact
            
        except Exception as e:
            logger.warning(f"Could not compute causal impact for {cause}: {e}")
            return 0.0
    
    def _find_causal_paths(
        self, 
        start: str, 
        end: str,
        max_length: int = 5
    ) -> List[CausalPath]:
        """
        Find all causal paths from start to end in the graph.
        
        A causal path shows the chain of causation:
        network_congestion → delay → cpu → slo_violated
        """
        if start not in self.causal_graph.nodes() or end not in self.causal_graph.nodes():
            return []
        
        all_paths = []
        
        def dfs(current: str, target: str, path: List[str], visited: Set[str]):
            """Depth-first search to find paths"""
            if len(path) > max_length:
                return
            
            if current == target:
                # Found a path!
                strength = self._compute_path_strength(path)
                confidence = self._compute_path_confidence(path)
                explanation = self._explain_path(path)
                
                all_paths.append(CausalPath(
                    path=path.copy(),
                    strength=strength,
                    confidence=confidence,
                    explanation=explanation
                ))
                return
            
            # Explore children (variables this causes)
            for child in self.causal_graph.successors(current):
                if child not in visited:
                    visited.add(child)
                    path.append(child)
                    dfs(child, target, path, visited)
                    path.pop()
                    visited.remove(child)
        
        dfs(start, end, [start], {start})
        
        # Sort by strength
        all_paths.sort(key=lambda p: p.strength, reverse=True)
        
        return all_paths
    
    def _compute_path_strength(self, path: List[str]) -> float:
        """
        Compute the strength of a causal path.
        
        Strength = Product of edge strengths along the path
        """
        if len(path) < 2:
            return 0.0
        
        strength = 1.0
        
        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i + 1]
            
            # Edge strength = How much parent explains child
            # We approximate this using the number of edges
            # (In reality, you'd use conditional mutual information)
            edge_strength = 0.8  # Default
            
            strength *= edge_strength
        
        return strength
    
    def _compute_path_confidence(self, path: List[str]) -> float:
        """
        Compute confidence in this causal path.
        
        Confidence decreases with path length (longer paths = more uncertainty)
        """
        base_confidence = 0.95
        decay = 0.1 * (len(path) - 1)
        
        return max(0.1, base_confidence - decay)
    
    def _explain_path(self, path: List[str]) -> str:
        """Generate human-readable explanation of causal path"""
        if len(path) < 2:
            return ""
        
        explanation_parts = []
        
        for i in range(len(path) - 1):
            cause = path[i].replace('_', ' ')
            effect = path[i + 1].replace('_', ' ')
            explanation_parts.append(f"{cause} causes {effect}")
        
        return " → ".join(explanation_parts)
    
    def _compute_evidence_strength(
        self, 
        variable: str, 
        observation: Dict[str, Any]
    ) -> float:
        """
        How strong is the evidence that this variable is problematic?
        
        Evidence strength = How far from normal is this value?
        """
        if variable not in observation:
            return 0.0
        
        value = observation[variable]
        
        # For numeric variables, check if it's an outlier
        if isinstance(value, (int, float)):
            # Heuristic: Values > 70 are problematic
            if value > 70:
                return min(1.0, value / 100.0)
            else:
                return 0.2
        
        # For boolean, True = problematic
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        
        # Discrete BN states (must match training / bn_discretize)
        if isinstance(value, str):
            v = value.lower()
            tables = {
                "cpu": {"high": 1.0, "normal": 0.12},
                "memory": {"high": 1.0, "normal": 0.12},
                "delay": {"high": 1.0, "normal": 0.12},
                "network_quality": {"low": 1.0, "medium": 0.6, "high": 0.12},
            }
            if variable in tables:
                return tables[variable].get(v, 0.45)
        
        return 0.5
    
    def _generate_explanation(
        self,
        cause_var: str,
        symptom: str,
        paths: List[CausalPath],
        observation: Dict[str, Any]
    ) -> str:
        """
        Generate a human-readable explanation of why this is a root cause.
        """
        value = observation.get(cause_var, 'unknown')
        
        # Build explanation
        explanation = f"Variable '{cause_var}' is a root cause because:\n"
        explanation += f"  - Current value: {value}\n"
        explanation += f"  - This causally influences '{symptom}' through {len(paths)} pathway(s):\n"
        
        for i, path in enumerate(paths[:3], 1):  # Show top 3 paths
            explanation += f"    {i}. {path.explanation}\n"
        
        return explanation
    
    def explain_violation(
        self, 
        observation: Dict[str, Any],
        symptom: str = 'slo_violated'
    ) -> str:
        """
        Generate a complete explanation of why an SLO violation occurred.
        
        This is the main output for operators/administrators.
        """
        if symptom not in observation or not observation[symptom]:
            return "No violation detected."
        
        # Find root causes
        root_causes = self.find_root_causes(
            symptom=symptom,
            symptom_value=observation[symptom],
            observation=observation,
            top_k=3
        )
        
        if not root_causes:
            return "Could not determine root cause (model may be incomplete)."
        
        # Generate report
        report = "=" * 70 + "\n"
        report += "ROOT CAUSE ANALYSIS\n"
        report += "=" * 70 + "\n\n"
        
        report += f"Symptom: {symptom} = {observation[symptom]}\n\n"
        
        report += "Identified Root Causes (ranked by impact):\n\n"
        
        for i, rc in enumerate(root_causes, 1):
            report += f"{i}. {rc}\n"
            report += f"   Evidence strength: {rc.evidence_strength:.2f}\n"
            report += f"   Causal pathways:\n"
            
            for path in rc.causal_paths[:2]:  # Show top 2 paths
                report += f"     • {path}\n"
            
            report += f"\n   Explanation:\n"
            for line in rc.explanation.split('\n'):
                report += f"     {line}\n"
            report += "\n"
        
        report += "=" * 70 + "\n"
        
        return report


class DistributedRootCauseAnalyzer:
    """
    Coordinates root cause analysis across multiple micro-agents.
    
    Each micro-agent (on IoT device, edge server, etc.) runs local analysis,
    then results are aggregated to find system-wide root causes.
    """
    
    def __init__(self):
        self.local_analyzers: Dict[str, RootCauseAnalyzer] = {}
        self.global_causes: List[RootCause] = []
    
    def register_agent(self, agent_id: str, analyzer: RootCauseAnalyzer):
        """Register a micro-agent's root cause analyzer"""
        self.local_analyzers[agent_id] = analyzer
        logger.info(f"Registered analyzer for agent {agent_id}")
    
    def aggregate_root_causes(
        self,
        local_observations: Dict[str, Dict[str, Any]],
        symptom: str = 'slo_violated'
    ) -> List[RootCause]:
        """
        Aggregate root causes from multiple micro-agents.
        
        Args:
            local_observations: {agent_id: observation_dict}
            symptom: The symptom to analyze
            
        Returns:
            Aggregated root causes across all agents
        """
        logger.info(f"Aggregating root causes from {len(local_observations)} agents")
        
        all_causes = defaultdict(list)
        
        # Collect root causes from each agent
        for agent_id, observation in local_observations.items():
            if agent_id not in self.local_analyzers:
                continue
            
            analyzer = self.local_analyzers[agent_id]
            
            if symptom not in observation:
                continue
            
            # Find local root causes
            local_causes = analyzer.find_root_causes(
                symptom=symptom,
                symptom_value=observation[symptom],
                observation=observation,
                top_k=5
            )
            
            # Group by variable
            for cause in local_causes:
                all_causes[cause.variable].append((agent_id, cause))
        
        # Aggregate: Find causes that appear across multiple agents
        global_causes = []
        
        for variable, agent_causes in all_causes.items():
            # Count how many agents identified this as a root cause
            num_agents = len(agent_causes)
            
            # Average impact across agents
            avg_impact = np.mean([cause.impact_score for _, cause in agent_causes])
            
            # Combine explanations
            combined_explanation = f"Root cause '{variable}' detected by {num_agents} agent(s):\n"
            for agent_id, cause in agent_causes:
                combined_explanation += f"  - Agent {agent_id}: impact={cause.impact_score:.2f}\n"
            
            # Create aggregated root cause
            global_cause = RootCause(
                variable=variable,
                value=agent_causes[0][1].value,  # Use first agent's value
                impact_score=avg_impact * (1 + 0.1 * num_agents),  # Boost if multiple agents agree
                causal_paths=agent_causes[0][1].causal_paths,
                evidence_strength=np.mean([cause.evidence_strength for _, cause in agent_causes]),
                explanation=combined_explanation
            )
            
            global_causes.append(global_cause)
        
        # Sort by impact
        global_causes.sort(key=lambda c: c.impact_score, reverse=True)
        
        logger.info(f"Found {len(global_causes)} global root causes")
        
        return global_causes


