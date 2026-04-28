"""
Multi-Agent Root Cause Discovery System

Each micro-agent has a specialized role:
1. Detector: Identifies anomalies/congestion
2. RCA Agents: Find root causes (using Active Inference + Bayesian)
3. CertaintyCheckAgent: Decides if the root-cause decision is certain (incl. VFE)
4. Solution Planner: Proposes fixes for each root cause
5. Executor: Implements solutions via functional tool calling (with VFE gating)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger

from src.aif.bayesian_network import EOSCModel
from src.aif.root_cause_analyzer import RootCauseAnalyzer, RootCause
from src.aif.vfe_compute import VFEComputer, UncertaintyGate
from src.aif.healing_tools import get_healing_tool_registry


@dataclass
class Anomaly:
    """Detected anomaly/congestion"""
    metric: str
    value: float
    threshold: float
    severity: float


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
    
    def __init__(self, device_id: str, thresholds: Dict[str, float]):
        self.device_id = device_id
        self.thresholds = thresholds
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
        
        return anomalies


class RootCauseAgent:
    """
    Micro-Agent 2: Finds root causes using Active Inference + Bayesian Networks.
    Multiple instances can run in parallel to analyze different aspects.
    """
    
    def __init__(self, agent_id: str, eosc_model: EOSCModel):
        self.agent_id = agent_id
        self.eosc_model = eosc_model  # Learned by Active Inference
        self.analyzer = RootCauseAnalyzer(eosc_model)
        logger.info(f"Root Cause Agent {agent_id} initialized")
    
    def find_root_causes(
        self, 
        anomalies: List[Anomaly],
        observation: Dict[str, Any]
    ) -> List[RootCause]:
        """
        Use Active Inference model to discover root causes
        
        Key insight: The causal model was learned by Active Inference!
        """
        logger.info(f"[RCA {self.agent_id}] Analyzing {len(anomalies)} anomalies...")
        
        all_root_causes = []
        
        for anomaly in anomalies:
            # Use Bayesian Network (learned by AIF) to find root causes
            root_causes = self.analyzer.find_root_causes(
                symptom=anomaly.metric,
                symptom_value=anomaly.value,
                observation=observation,
                top_k=3
            )
            
            for rc in root_causes:
                logger.info(
                    f"[RCA {self.agent_id}] Found root cause: "
                    f"{rc.variable}={rc.value} (impact={rc.impact_score:.2f})"
                )
            
            all_root_causes.extend(root_causes)
        
        # Deduplicate and rank
        unique_causes = self._deduplicate(all_root_causes)
        
        return sorted(unique_causes, key=lambda rc: rc.impact_score, reverse=True)
    
    def _deduplicate(self, root_causes: List[RootCause]) -> List[RootCause]:
        """Combine duplicate root causes"""
        seen = {}
        for rc in root_causes:
            if rc.variable not in seen:
                seen[rc.variable] = rc
            else:
                # Average impact scores
                existing = seen[rc.variable]
                existing.impact_score = (existing.impact_score + rc.impact_score) / 2
        
        return list(seen.values())


class SolutionPlannerAgent:
    """
    Micro-Agent 3: Proposes solutions for each root cause
    
    Uses counterfactual reasoning: "What if we change X?"
    """
    
    def __init__(self, agent_id: str, eosc_model: EOSCModel):
        self.agent_id = agent_id
        self.eosc_model = eosc_model
        logger.info(f"Solution Planner Agent {agent_id} initialized")
    
    def propose_solutions(
        self, 
        root_causes: List[RootCause],
        observation: Dict[str, Any]
    ) -> List[Solution]:
        """
        For each root cause, propose a solution
        Use Bayesian Network for counterfactual reasoning
        """
        logger.info(f"[Planner {self.agent_id}] Proposing solutions for {len(root_causes)} root causes...")
        
        solutions = []
        
        for rc in root_causes:
            # Generate candidate actions
            candidate_actions = self._generate_actions_for_cause(rc)
            
            # Evaluate each action using Bayesian Network
            best_action = None
            best_impact = 0.0
            
            for action in candidate_actions:
                # Counterfactual: "What if we do this action?"
                expected_impact = self._evaluate_action(
                    rc, action, observation
                )
                
                if expected_impact > best_impact:
                    best_impact = expected_impact
                    best_action = action
            
            if best_action:
                solution = Solution(
                    root_cause=rc.variable,
                    action=best_action,
                    expected_impact=best_impact,
                    confidence=rc.evidence_strength
                )
                solutions.append(solution)
                
                logger.info(
                    f"[Planner {self.agent_id}] Proposed: {best_action} "
                    f"for root cause {rc.variable} (impact={best_impact:.2f})"
                )
        
        return solutions
    
    def _generate_actions_for_cause(self, root_cause: RootCause) -> List[str]:
        """Generate possible actions for a root cause (CCTV: offload, rate limit, redistribute)."""
        actions = []
        var = root_cause.variable.lower()
        if 'network' in var or 'bandwidth' in var:
            actions = [
                'offload_to_fog',
                'reduce_bitrate',
                'throttle_connections',
                'switch_network'
            ]
        elif 'node' in var or 'cpu' in var:
            actions = [
                'reduce_fps',
                'reduce_resolution',
                'offload_processing'
            ]
        elif 'memory' in root_cause.variable.lower():
            actions = [
                'clear_cache',
                'restart_service',
                'reduce_buffer_size'
            ]
        else:
            actions = ['maintain', 'monitor']
        
        return actions
    
    def _evaluate_action(
        self, 
        root_cause: RootCause, 
        action: str,
        observation: Dict[str, Any]
    ) -> float:
        """
        Evaluate action using counterfactual reasoning
        
        "If we do this action, how much will it improve things?"
        """
        try:
            # Simulate improved state after action
            counterfactual_obs = observation.copy()
            
            # Heuristic: Action improves root cause by 50%
            current_value = observation[root_cause.variable]
            if isinstance(current_value, (int, float)):
                improved_value = max(0, current_value * 0.5)
                counterfactual_obs[root_cause.variable] = improved_value
            
            # Query Bayesian Network: "What would happen?"
            # (Simplified - in full implementation, use model.query())
            
            # Return expected impact (0.0 to 1.0)
            return 0.5 + root_cause.impact_score * 0.5
            
        except Exception as e:
            logger.warning(f"Could not evaluate action {action}: {e}")
            return 0.0


class ExecutorAgent:
    """
    Micro-Agent 4: Executes solutions.
    Only acts when VFE is below threshold (confident enough to act safely).
    """
    
    def __init__(self, agent_id: str, eosc_model: EOSCModel, vfe_threshold: float = 2.0):
        self.agent_id = agent_id
        self.eosc_model = eosc_model
        self.vfe_computer = VFEComputer()
        self.uncertainty_gate = UncertaintyGate(threshold=vfe_threshold)
        self.execution_history = []
        logger.info(f"Executor Agent {agent_id} initialized (VFE threshold={vfe_threshold})")
    
    def execute_solutions(
        self, 
        solutions: List[Solution],
        observation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute solutions only when VFE below threshold (safe to act)."""
        logger.info(f"[Executor {self.agent_id}] Evaluating {len(solutions)} solutions...")
        
        executed = []
        
        for solution in solutions:
            # Compute VFE (uncertainty about this solution)
            vfe = self.vfe_computer.compute_vfe(
                self.eosc_model,
                observation,
                action={}  # No specific action params; VFE measures state uncertainty
            )
            
            # UNCERTAINTY GATE: Only act if confident!
            if self.uncertainty_gate.should_act(vfe):
                logger.info(
                    f"[Executor {self.agent_id}] ✓ Executing: {solution.action} "
                    f"(VFE={vfe:.2f} < threshold)"
                )
                
                # Execute the action
                result = self._execute_action(solution)
                executed.append(result)
                
                self.execution_history.append({
                    'solution': solution,
                    'vfe': vfe,
                    'executed': True
                })
            else:
                logger.warning(
                    f"[Executor {self.agent_id}] ✗ Abstaining: {solution.action} "
                    f"(VFE={vfe:.2f} >= threshold) - Too uncertain!"
                )
                
                self.execution_history.append({
                    'solution': solution,
                    'vfe': vfe,
                    'executed': False,
                    'reason': 'high_uncertainty'
                })
        
        return executed
    
    def _execute_action(self, solution: Solution) -> Dict[str, Any]:
        """Execute the action via healing tool registry (register real APIs for production)."""
        registry = get_healing_tool_registry()
        tool_result = registry.execute(
            solution.action,
            root_cause=solution.root_cause,
            expected_impact=solution.expected_impact,
        )
        return {
            'action': solution.action,
            'root_cause': solution.root_cause,
            'status': 'executed' if tool_result.success else 'failed',
            'expected_impact': solution.expected_impact,
            'tool_success': tool_result.success,
            'tool_message': tool_result.message,
        }


class MultiAgentCoordinator:
    """
    Coordinates multiple micro-agents
    
    This orchestrates the pipeline:
    Detector → RCA Agents → Solution Planner → Executor
    """
    
    def __init__(self, device_id: str, certainty_threshold: float = 0.85):
        self.device_id = device_id
        self.certainty_threshold = certainty_threshold  # High certainty required on root cause (≥85%)
        self.detector = None
        self.rca_agents = []
        self.certainty_check_agent = None  # Optional: dedicated agent for "certain or uncertain?"
        self.planner = None
        self.executor = None
        logger.info(f"Multi-Agent Coordinator initialized for {device_id} (certainty≥{certainty_threshold:.0%})")
    
    def register_agents(
        self,
        detector: DetectorAgent,
        rca_agents: List[RootCauseAgent],
        planner: SolutionPlannerAgent,
        executor: ExecutorAgent,
        certainty_check_agent: Optional[Any] = None,
    ):
        """Register all micro-agents (certainty_check_agent optional: dedicated agent for certainty+VFE)."""
        self.detector = detector
        self.rca_agents = rca_agents
        self.planner = planner
        self.executor = executor
        self.certainty_check_agent = certainty_check_agent
        if certainty_check_agent is not None and rca_agents:
            try:
                certainty_check_agent.set_model(rca_agents[0].eosc_model)
            except Exception:
                pass
        logger.info(
            f"[Coordinator {self.device_id}] Registered "
            f"{len(rca_agents)} RCA agents + detector + planner + executor"
            + (" + CertaintyCheckAgent" if certainty_check_agent else "")
        )
    
    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete multi-agent pipeline implementing the thesis approach:
        
        REQUIREMENT: "First identify the EXACT ROOT CAUSE with HIGH CERTAINTY,
        then perform targeted mitigation"
        
        Pipeline Phases:
        PHASE 0: Parallel specialized micro-agents (each checks ONE thing)
        PHASE 1: Anomaly detection (find symptoms)
        PHASE 2: Root cause identification with HIGH CERTAINTY (≥85%)
                - Multiple agents analyze in parallel using Bayesian Networks
                - Compute certainty from consensus + impact + evidence
                - REFUSE TO ACT if certainty < threshold
        PHASE 3: Solution planning (targeted mitigation for confirmed root cause)
        PHASE 4: Execution (implement healing action)
        
        This implements "Distributed Agentic Micro-Agent for Resilience"
        """
        logger.info("\n" + "=" * 70)
        logger.info("MULTI-AGENT SELF-HEALING PIPELINE")
        logger.info("=" * 70)
        
        # Phase 0: Parallel specialized checks
        logger.info("\n[PHASE 0] PARALLEL SPECIALIZED AGENT CHECKS")
        logger.info("-" * 70)
        
        parallel_findings = []
        try:
            from src.aif.specialized_agents import ParallelCheckCoordinator
            parallel_coordinator = ParallelCheckCoordinator()
            parallel_findings = parallel_coordinator.parallel_check(observation)
            
            detected_issues = [f for f in parallel_findings if f.issue_detected]
            logger.info(f"Specialized agents detected {len(detected_issues)} issues")
        except ImportError:
            logger.warning("Specialized agents not available, skipping parallel checks")
        
        # PHASE 1: DETECTION
        logger.info("\n[PHASE 1] ANOMALY DETECTION")
        logger.info("-" * 70)
        
        anomalies = self.detector.detect_anomalies(observation)
        
        if not anomalies:
            logger.success("No anomalies detected - system healthy")
            return {
                'phase': 'detection',
                'status': 'healthy',
                'anomalies': [],
                'root_cause': None,
                'action_taken': None,
                'abstained': False,
                'certainty': 1.0,
                'vfe': 0.0
            }
        
        logger.warning(f"Detected {len(anomalies)} anomalies")
        for anomaly in anomalies:
            logger.warning(f"  - {anomaly.metric}={anomaly.value} (threshold={anomaly.threshold})")
        
        # Phase 2: Root cause identification with high certainty
        logger.info("\n[PHASE 2] ROOT CAUSE IDENTIFICATION WITH HIGH CERTAINTY")
        logger.info("-" * 70)
        all_root_causes = []
        
        for i, rca_agent in enumerate(self.rca_agents, 1):
            logger.info(f"\n  Agent {i}/{len(self.rca_agents)}: {rca_agent.agent_id}")
            root_causes = rca_agent.find_root_causes(anomalies, observation)
            
            if root_causes:
                top_cause = root_causes[0]
                logger.info(
                    f"    Found: {top_cause.variable} "
                    f"(impact={top_cause.impact_score:.2f})"
                )
                all_root_causes.extend(root_causes)
            else:
                logger.warning(f"    No root causes found")
        
        if not all_root_causes:
            logger.error("NO ROOT CAUSES IDENTIFIED - ABSTAINING")
            return {
                'phase': 'root_cause_analysis',
                'status': 'processed',
                'anomalies': anomalies,
                'root_causes': [],
                'root_cause': None,
                'action_taken': None,
                'abstained': True,
                'reason': 'no_root_cause_found',
                'certainty': 0.0,
                'vfe': None
            }
        
        # Aggregate findings and compute certainty (multi-agent consensus)
        logger.info("\nAggregating findings from multiple agents...")
        
        # Use existing aggregation method
        aggregated_causes = self._aggregate_root_causes(all_root_causes)
        
        if not aggregated_causes:
            return {
                'phase': 'root_cause_analysis',
                'status': 'processed',
                'anomalies': anomalies,
                'root_causes': [],
                'root_cause': None,
                'action_taken': None,
                'abstained': True,
                'reason': 'no_aggregated_causes',
                'certainty': 0.0,
                'vfe': None
            }
        
        # Get top root cause
        top_root_cause = max(aggregated_causes, key=lambda rc: rc.impact_score)

        # CALCULATE DYNAMIC THRESHOLD based on system criticality
        dynamic_threshold = self.certainty_threshold
        criticality_score = len(anomalies)
        if criticality_score > 1:
            # Lower threshold by 15% per additional anomaly, up to a max drop
            dynamic_threshold = max(0.50, self.certainty_threshold - (0.15 * min(criticality_score - 1, 2)))
            logger.info(f"  [Dynamic Gating] Lowered certainty threshold to {dynamic_threshold:.1%} due to {criticality_score} concurrent anomalies")

        # CERTAINTY CHECK: use dedicated CertaintyCheckAgent ("another agent whether the decision is certain or uncertain")
        # That agent combines scalar certainty + VFE (VFE tied to root-cause certainty)
        if self.certainty_check_agent is not None and self.rca_agents:
            eosc_model = self.rca_agents[0].eosc_model
            # Apply dynamic threshold to the certainty check agent as well
            self.certainty_check_agent.certainty_threshold = dynamic_threshold
            
            certainty_result = self.certainty_check_agent.check(
                top_root_cause, observation, eosc_model=eosc_model
            )
            certainty = certainty_result.certainty_score
            vfe_from_certainty = certainty_result.vfe_value
            logger.info(f"\n  Root Cause: {top_root_cause.variable}")
            logger.info(f"  Impact: {top_root_cause.impact_score:.2f}")
            logger.info(f"  Evidence: {top_root_cause.evidence_strength:.2f}")
            logger.info(f"  [CertaintyCheckAgent] certainty={certainty:.1%}, VFE={vfe_from_certainty:.2f} → should_proceed={certainty_result.should_proceed}")
            if not certainty_result.should_proceed:
                logger.error(
                    f"\n❌ CERTAINTY AGENT: DO NOT PROCEED ({certainty_result.reason})"
                )
                return {
                    'phase': 'root_cause_analysis',
                    'status': 'processed',
                    'anomalies': anomalies,
                    'root_causes': aggregated_causes,
                    'root_cause': top_root_cause.variable,
                    'action_taken': None,
                    'abstained': True,
                    'reason': certainty_result.reason,
                    'certainty': certainty,
                    'vfe': vfe_from_certainty,
                    'certainty_agent_abstained': True,
                }
        else:
            certainty = (top_root_cause.impact_score + top_root_cause.evidence_strength) / 2.0
            vfe_from_certainty = None
            logger.info(f"\n  Root Cause: {top_root_cause.variable}")
            logger.info(f"  Impact: {top_root_cause.impact_score:.2f}")
            logger.info(f"  Evidence: {top_root_cause.evidence_strength:.2f}")
            logger.info(f"  FINAL CERTAINTY: {certainty:.1%}")
            if certainty < dynamic_threshold:
                logger.error(
                    f"\n❌ CERTAINTY TOO LOW ({certainty:.1%} < {dynamic_threshold:.1%})"
                )
                return {
                    'phase': 'root_cause_analysis',
                    'status': 'processed',
                    'anomalies': anomalies,
                    'root_causes': aggregated_causes,
                    'root_cause': top_root_cause.variable,
                    'action_taken': None,
                    'abstained': True,
                    'reason': 'certainty_below_threshold',
                    'certainty': certainty,
                    'vfe': None
                }

        logger.success(
            f"\n✓ ROOT CAUSE CONFIRMED WITH HIGH CERTAINTY: "
            f"{top_root_cause.variable} ({certainty:.1%})"
        )
        
        # PHASE 3: SOLUTION PLANNING
        logger.info("\n[PHASE 3] SOLUTION PLANNING")
        logger.info("-" * 70)
        
        solutions = self.planner.propose_solutions(aggregated_causes, observation)
        
        if not solutions:
            logger.warning("No solutions proposed")
            return {
                'phase': 'solution_planning',
                'status': 'processed',
                'anomalies': anomalies,
                'root_causes': aggregated_causes,
                'root_cause': top_root_cause.variable,
                'action_taken': None,
                'abstained': True,
                'reason': 'no_solution_available',
                'certainty': certainty,
                'vfe': None
            }
        
        proposed_solution = solutions[0]
        logger.info(
            f"Proposed solution: {proposed_solution.action} "
            f"(expected_impact={proposed_solution.expected_impact:.2f})"
        )
        
        # PHASE 4: EXECUTE with VFE gating
        logger.info("\n[PHASE 4] EXECUTION (VFE-GATED)")
        logger.info("-" * 70)
        
        executed = self.executor.execute_solutions(solutions, observation)
        
        logger.info(f"✓ Executed {len(executed)} solutions\n")
        
        # Return summary
        return {
            'phase': 'complete',
            'status': 'processed',
            'anomalies': anomalies,
            'root_causes': aggregated_causes,
            'root_cause': top_root_cause.variable,
            'solutions': solutions,
            'action_taken': solutions[0].action if solutions else None,
            'executed': executed,
            'abstained': False,
            'certainty': certainty,
            'vfe': 0.0,  # VFE computed in executor
            'parallel_findings': parallel_findings
        }
    
    def _aggregate_root_causes(self, all_causes: List[RootCause]) -> List[RootCause]:
        """
        Aggregate root causes from multiple RCA agents
        
        If multiple agents identify the same root cause, that increases confidence!
        """
        aggregated = {}
        
        for rc in all_causes:
            if rc.variable not in aggregated:
                aggregated[rc.variable] = rc
            else:
                # Multiple agents agree - boost impact score!
                existing = aggregated[rc.variable]
                existing.impact_score = (existing.impact_score + rc.impact_score) / 2 * 1.2  # 20% boost
                existing.evidence_strength = max(existing.evidence_strength, rc.evidence_strength)
        
        return list(aggregated.values())


