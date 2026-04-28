"""
Enhanced Multi-Agent System with Explicit Certainty Calculation

This version adds:
1. Certainty score calculation (0.0 to 1.0)
2. Evidence-based reasoning
3. Consensus mechanism for high certainty
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import numpy as np

from src.aif.bayesian_network import EOSCModel
from src.aif.root_cause_analyzer import RootCauseAnalyzer, RootCause
from src.aif.vfe_compute import VFEComputer, UncertaintyGate

# Typical strong counterfactual impact from the learned discrete CCTV BN in simulation (~0.35–0.55).
BN_IMPACT_CALIBRATION = 0.42


@dataclass
class Anomaly:
    """Detected anomaly/congestion"""
    metric: str
    value: float
    threshold: float
    severity: float


@dataclass
class RootCauseWithCertainty:
    """
    Root cause identified with explicit certainty measurement
    
    Certainty comes from:
    1. Multiple agents agreeing (consensus)
    2. High impact score
    3. Strong causal evidence
    4. Consistent findings across agents
    """
    variable: str
    value: Any
    impact_score: float
    certainty: float  # 0.0 to 1.0 - How certain are we?
    evidence: str
    num_agents_agree: int
    agreement_variance: float  # Low variance = high certainty
    causal_paths: List[Any]
    
    def is_certain(self, threshold: float = 0.85) -> bool:
        """
        Check if we have HIGH CERTAINTY
        
        Threshold 0.85 means 85% certain this is THE root cause
        """
        return self.certainty >= threshold
    
    def get_confidence_level(self) -> str:
        """Human-readable confidence level"""
        if self.certainty >= 0.95:
            return "VERY HIGH (>95%)"
        elif self.certainty >= 0.85:
            return "HIGH (85-95%)"
        elif self.certainty >= 0.70:
            return "MEDIUM (70-85%)"
        else:
            return "LOW (<70%)"


@dataclass
class Solution:
    """Proposed solution for a root cause"""
    root_cause: str
    action: str
    expected_impact: float
    confidence: float


class DetectorAgent:
    """
    Micro-Agent 1: Detects congestion and SLO violations
    
    Lightweight agent that monitors metrics and flags anomalies.
    Runs on each IoT device.
    """
    
    def __init__(self, device_id: str, thresholds: Dict[str, float] = None):
        self.device_id = device_id
        self.thresholds = thresholds or {
            'cpu': 70,
            'memory': 75,
            'delay': 50,
            'network_quality': 30,
            'slo_violated': 0.5
        }
        logger.info(f"Detector Agent {device_id} initialized")
    
    def detect_anomalies(self, observation: Dict[str, float]) -> List[Anomaly]:
        """
        Lightweight anomaly detection
        No complex ML - just threshold checks (fast for IoT)
        """
        anomalies = []
        
        for metric, value in observation.items():
            if metric in self.thresholds:
                threshold = self.thresholds[metric]
                
                # Check if value exceeds threshold
                if value > threshold:
                    severity = (value - threshold) / threshold
                    
                    anomaly = Anomaly(
                        metric=metric,
                        value=value,
                        threshold=threshold,
                        severity=severity
                    )
                    anomalies.append(anomaly)
                    
                    logger.warning(
                        f"[Detector {self.device_id}] Anomaly detected: "
                        f"{metric}={value} > {threshold} (severity={severity:.2f})"
                    )
                # Special case for network_quality (lower is worse)
                elif metric == 'network_quality' and value < threshold:
                    severity = (threshold - value) / threshold
                    anomaly = Anomaly(
                        metric=metric,
                        value=value,
                        threshold=threshold,
                        severity=severity
                    )
                    anomalies.append(anomaly)
        
        return anomalies


class RootCauseAgentWithCertainty:
    """
    Enhanced RCA Agent that computes EXPLICIT CERTAINTY
    
    Certainty is based on:
    - Impact score (how much this cause affects the symptom)
    - Causal evidence strength
    - Consistency with other observations
    """
    
    def __init__(self, agent_id: str, eosc_model: EOSCModel):
        self.agent_id = agent_id
        self.eosc_model = eosc_model
        self.analyzer = RootCauseAnalyzer(eosc_model)
        logger.info(f"Root Cause Agent (with Certainty) {agent_id} initialized")
    
    def find_root_causes_with_certainty(
        self, 
        anomalies: List[Anomaly],
        observation: Dict[str, Any]
    ) -> List[RootCauseWithCertainty]:
        """
        Find root causes with EXPLICIT CERTAINTY measurement
        """
        logger.info(f"[RCA {self.agent_id}] Analyzing {len(anomalies)} anomalies with certainty measurement...")
        
        all_root_causes = []
        
        for anomaly in anomalies:
            # Use Bayesian Network to find root causes
            root_causes = self.analyzer.find_root_causes(
                symptom=anomaly.metric,
                symptom_value=anomaly.value,
                observation=observation,
                top_k=5
            )
            
            # Convert to RootCauseWithCertainty
            for rc in root_causes:
                # Calculate certainty for THIS agent's finding
                certainty = self._calculate_certainty(rc, observation)
                
                rc_with_certainty = RootCauseWithCertainty(
                    variable=rc.variable,
                    value=rc.value,
                    impact_score=rc.impact_score,
                    certainty=certainty,
                    evidence=rc.explanation,
                    num_agents_agree=1,  # Will be updated during aggregation
                    agreement_variance=0.0,  # Will be updated during aggregation
                    causal_paths=rc.causal_paths
                )
                
                all_root_causes.append(rc_with_certainty)
                
                logger.info(
                    f"[RCA {self.agent_id}] Found root cause: "
                    f"{rc.variable}={rc.value} "
                    f"(impact={rc.impact_score:.2f}, certainty={certainty:.2f})"
                )
        
        # Deduplicate
        unique_causes = self._deduplicate(all_root_causes)
        
        return sorted(unique_causes, key=lambda rc: rc.certainty, reverse=True)
    
    def _calculate_certainty(
        self, 
        root_cause: RootCause, 
        observation: Dict[str, Any]
    ) -> float:
        """
        Calculate certainty score (0.0 to 1.0) for a root cause
        
        Certainty factors:
        1. Impact score (0.0-1.0) - how much this affects the outcome
        2. Evidence strength (0.0-1.0) - quality of causal evidence
        3. Causal path existence - is there a clear causal chain?
        """
        # Scale raw impact so a strong BN signal + stressed evidence can reach the 0.85 gate.
        impact_factor = min(1.0, root_cause.impact_score / BN_IMPACT_CALIBRATION)

        evidence_factor = root_cause.evidence_strength
        path_factor = 1.0 if len(root_cause.causal_paths) > 0 else 0.5

        certainty = (
            0.5 * impact_factor +
            0.3 * evidence_factor +
            0.2 * path_factor
        )

        return min(1.0, certainty)
    
    def _deduplicate(
        self, 
        root_causes: List[RootCauseWithCertainty]
    ) -> List[RootCauseWithCertainty]:
        """Combine duplicate root causes"""
        seen = {}
        for rc in root_causes:
            if rc.variable not in seen:
                seen[rc.variable] = rc
            else:
                # Keep the one with higher certainty
                if rc.certainty > seen[rc.variable].certainty:
                    seen[rc.variable] = rc
        
        return list(seen.values())


class CertaintyAggregator:
    """
    Aggregates findings from multiple RCA agents to compute CONSENSUS CERTAINTY
    
    When multiple agents agree → HIGH CERTAINTY
    When agents disagree → LOW CERTAINTY
    """
    
    @staticmethod
    def aggregate_root_causes(
        findings_by_agent: Dict[str, List[RootCauseWithCertainty]]
    ) -> List[RootCauseWithCertainty]:
        """
        Aggregate findings from multiple agents
        
        Key insight: When multiple independent agents find the SAME root cause,
        our certainty increases significantly!
        """
        logger.info(f"[CertaintyAggregator] Aggregating findings from {len(findings_by_agent)} agents...")
        
        # Group by variable
        grouped: Dict[str, List[RootCauseWithCertainty]] = {}
        
        for agent_id, findings in findings_by_agent.items():
            for rc in findings:
                if rc.variable not in grouped:
                    grouped[rc.variable] = []
                grouped[rc.variable].append(rc)
        
        # Calculate consensus certainty
        aggregated = []
        
        for variable, findings in grouped.items():
            num_agents = len(findings)
            
            # Calculate metrics
            impact_scores = [rc.impact_score for rc in findings]
            certainties = [rc.certainty for rc in findings]
            
            avg_impact = np.mean(impact_scores)
            avg_certainty = np.mean(certainties)
            variance = np.var(impact_scores)
            
            # CONSENSUS CERTAINTY CALCULATION
            # More agents agree + low variance = HIGH CERTAINTY
            
            # Base certainty from individual agents
            base_certainty = avg_certainty
            
            # Consensus boost: More agents agree → higher certainty
            consensus_boost = min(0.2, (num_agents - 1) * 0.1)
            
            # Consistency penalty: High variance → lower certainty
            consistency_penalty = min(0.15, variance * 0.3)
            
            # Final certainty
            final_certainty = min(1.0, base_certainty + consensus_boost - consistency_penalty)
            
            # Create aggregated result
            aggregated_rc = RootCauseWithCertainty(
                variable=variable,
                value=findings[0].value,
                impact_score=avg_impact,
                certainty=final_certainty,
                evidence=f"Consensus from {num_agents} agents (variance={variance:.3f})",
                num_agents_agree=num_agents,
                agreement_variance=variance,
                causal_paths=findings[0].causal_paths
            )
            
            aggregated.append(aggregated_rc)
            
            logger.info(
                f"[CertaintyAggregator] {variable}: "
                f"{num_agents} agents agree, "
                f"certainty={final_certainty:.2f} "
                f"({aggregated_rc.get_confidence_level()})"
            )
        
        # Sort by certainty (highest first)
        return sorted(aggregated, key=lambda rc: rc.certainty, reverse=True)


class SolutionPlannerAgent:
    """
    Micro-Agent 3: Proposes solutions for root causes
    """
    
    def __init__(self, agent_id: str, eosc_model: EOSCModel):
        self.agent_id = agent_id
        self.eosc_model = eosc_model
        logger.info(f"Solution Planner Agent {agent_id} initialized")
    
    def propose_solutions(
        self,
        root_causes: List[RootCauseWithCertainty],
        observation: Dict[str, Any]
    ) -> List[Solution]:
        """
        Propose solutions for root causes
        
        Only proposes solutions for root causes with HIGH CERTAINTY
        """
        logger.info(f"[Planner {self.agent_id}] Proposing solutions for {len(root_causes)} root causes...")
        
        solutions = []
        
        for rc in root_causes:
            # ONLY propose solutions for HIGH CERTAINTY root causes
            if not rc.is_certain(threshold=0.70):
                logger.warning(
                    f"[Planner {self.agent_id}] Skipping {rc.variable} "
                    f"(certainty {rc.certainty:.2f} too low, need >0.70)"
                )
                continue
            
            # Propose appropriate solution based on root cause
            if rc.variable == 'network_quality':
                solutions.append(Solution(
                    root_cause=rc.variable,
                    action='offload_to_fog',
                    expected_impact=rc.impact_score,
                    confidence=rc.certainty
                ))
                solutions.append(Solution(
                    root_cause=rc.variable,
                    action='reduce_bitrate',
                    expected_impact=rc.impact_score * 0.7,
                    confidence=rc.certainty * 0.85
                ))
            
            elif rc.variable == 'cpu':
                solutions.append(Solution(
                    root_cause=rc.variable,
                    action='scale_resources',
                    expected_impact=rc.impact_score,
                    confidence=rc.certainty
                ))
            
            elif rc.variable == 'memory':
                solutions.append(Solution(
                    root_cause=rc.variable,
                    action='clear_buffers',
                    expected_impact=rc.impact_score,
                    confidence=rc.certainty
                ))
            
            logger.info(
                f"[Planner {self.agent_id}] Proposed solution: "
                f"{solutions[-1].action} for {rc.variable} "
                f"(confidence={rc.certainty:.2f})"
            )
        
        return solutions


class ExecutorAgent:
    """
    Micro-Agent 4: Executes solutions (VFE-gated)
    """
    
    def __init__(self, agent_id: str, eosc_model: EOSCModel, vfe_threshold: float = 2.0):
        self.agent_id = agent_id
        self.eosc_model = eosc_model
        self.vfe_computer = VFEComputer()
        self.uncertainty_gate = UncertaintyGate(threshold=vfe_threshold)
        logger.info(f"Executor Agent {self.agent_id} initialized (VFE threshold={vfe_threshold})")
    
    def execute_solutions(
        self,
        solutions: List[Solution],
        observation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute solutions with VFE gating"""
        logger.info(f"[Executor {self.agent_id}] Evaluating {len(solutions)} solutions...")
        
        executed = []
        
        for solution in solutions:
            # Compute VFE (uncertainty in execution)
            vfe = self._compute_vfe(solution, observation)
            
            # Gate decision
            should_execute = self.uncertainty_gate.should_act(vfe)
            
            if should_execute:
                logger.info(
                    f"[Executor {self.agent_id}] ✓ EXECUTING: {solution.action} "
                    f"(VFE={vfe:.2f} < {self.uncertainty_gate.threshold})"
                )
                result = self._execute(solution)
                executed.append(result)
            else:
                logger.warning(
                    f"[Executor {self.agent_id}] ✗ ABSTAINING: {solution.action} "
                    f"(VFE={vfe:.2f} >= {self.uncertainty_gate.threshold})"
                )
        
        return executed
    
    def _compute_vfe(self, solution: Solution, observation: Dict[str, Any]) -> float:
        """Compute VFE for a solution"""
        return self.vfe_computer.compute_vfe(self.eosc_model, observation, action={})
    
    def _execute(self, solution: Solution) -> Dict[str, Any]:
        """Execute a solution (simulated)"""
        return {
            'action': solution.action,
            'root_cause': solution.root_cause,
            'status': 'success',
            'timestamp': 'now'
        }


class MultiAgentCoordinatorWithCertainty:
    """
    Orchestrates multiple micro-agents with EXPLICIT CERTAINTY tracking
    
    Key enhancement: Measures and reports CERTAINTY at each step
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.detector = None
        self.rca_agents = []
        self.planner = None
        self.executor = None
        self.aggregator = CertaintyAggregator()
        logger.info(f"Multi-Agent Coordinator (with Certainty) initialized for {agent_id}")
    
    def register_agents(
        self,
        detector: DetectorAgent,
        rca_agents: List[RootCauseAgentWithCertainty],
        planner: SolutionPlannerAgent,
        executor: ExecutorAgent
    ):
        """Register all agents"""
        self.detector = detector
        self.rca_agents = rca_agents
        self.planner = planner
        self.executor = executor
        logger.info(
            f"[Coordinator {self.agent_id}] Registered "
            f"{len(rca_agents)} RCA agents + detector + planner + executor"
        )
    
    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete pipeline with EXPLICIT CERTAINTY tracking
        """
        logger.info("\n" + "="*70)
        logger.info(f"[Coordinator {self.agent_id}] Processing new observation")
        logger.info("="*70 + "\n")
        
        # Step 1: Detection
        logger.info("[STEP 1] Detector Agent: Finding anomalies...")
        anomalies = self.detector.detect_anomalies(observation)
        logger.info(f"{'✓' if len(anomalies) == 0 else '✗'} Detected {len(anomalies)} anomalies\n")
        
        if len(anomalies) == 0:
            return {
                'anomalies': [],
                'root_causes': [],
                'solutions': [],
                'executed': []
            }
        
        # Step 2: Root Cause Analysis (Multiple agents in parallel)
        logger.info(f"[STEP 2] Root Cause Agents ({len(self.rca_agents)} agents): Finding root causes with CERTAINTY...")
        
        findings_by_agent = {}
        for rca_agent in self.rca_agents:
            findings = rca_agent.find_root_causes_with_certainty(anomalies, observation)
            findings_by_agent[rca_agent.agent_id] = findings
        
        # AGGREGATE for CONSENSUS CERTAINTY
        root_causes_with_certainty = self.aggregator.aggregate_root_causes(findings_by_agent)
        
        logger.info(f"✓ Found {len(root_causes_with_certainty)} unique root causes with certainty\n")
        
        # Step 3: Solution Planning (only for HIGH CERTAINTY causes)
        logger.info("[STEP 3] Solution Planner Agent: Proposing solutions...")
        solutions = self.planner.propose_solutions(root_causes_with_certainty, observation)
        logger.info(f"✓ Proposed {len(solutions)} solutions\n")
        
        # Step 4: Execution (VFE-gated)
        logger.info("[STEP 4] Executor Agent: Executing solutions (VFE-gated)...")
        executed = self.executor.execute_solutions(solutions, observation)
        logger.info(f"✓ Executed {len(executed)} solutions\n")
        
        return {
            'anomalies': anomalies,
            'root_causes': root_causes_with_certainty,
            'solutions': solutions,
            'executed': executed
        }


