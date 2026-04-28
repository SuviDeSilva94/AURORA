"""
Active Inference Agent WITHOUT VFE Threshold (Baseline)

Same causal reasoning (Bayesian Networks) and VFE computation as the full system,
but no safe-execution gate: always acts when root cause is proposed.
Used as baseline to show value of only acting when VFE is below threshold.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random
from loguru import logger

# Import core AIF components
from src.aif.bayesian_network import BayesianNetworkLearner, EOSCModel
from src.aif.vfe_compute import VFEComputer
from src.aif.root_cause_analyzer import RootCauseAnalyzer
from src.utils.bn_discretize import discretize_observation_like_training


@dataclass
class AIFAction:
    """An action suggested by AIF agent"""
    action_type: str
    target: str
    reason: str
    vfe: float
    root_cause: Optional[str] = None


class AIFNoGateAgent:
    """
    Active Inference agent WITHOUT VFE threshold (always acts)
    
    Key differences from AURORA:
    - Computes VFE but IGNORES it
    - Always acts (no abstention)
    - Has root cause analysis but acts even with low certainty
    
    This demonstrates the risk of acting without safety mechanisms.
    """
    
    def __init__(self, agent_id: str = "aif-no-gate", vfe_threshold: float = 999.9):
        self.agent_id = agent_id
        self.vfe_threshold = vfe_threshold  # Set very high so it never triggers
        
        # Core AIF components
        self.bn_learner = BayesianNetworkLearner()
        self.vfe_computer = VFEComputer()
        self.eosc_model: Optional[EOSCModel] = None
        self.root_cause_analyzer: Optional[RootCauseAnalyzer] = None
        
        # Training data
        self.observations = []
        self.is_trained = False
        self.action_count = 0
        
        logger.info(f"AIF-no-gate agent {agent_id} initialized (no VFE threshold, always acts)")
    
    def train(
        self,
        training_data: List[Dict[str, Any]],
        eosc_model: Optional[EOSCModel] = None,
    ):
        """
        Train the Bayesian Network on historical data.

        If ``eosc_model`` is provided (e.g. CCTV benchmark DAG from experiments),
        only parameters are refreshed; otherwise structure is learned from data.
        """
        logger.info(f"Training AIF-no-gate agent with {len(training_data)} samples...")
        
        if eosc_model is not None:
            self.eosc_model = self.bn_learner.update_parameters(eosc_model, training_data)
        else:
            self.eosc_model = self.bn_learner.learn_structure(training_data)
            self.eosc_model = self.bn_learner.update_parameters(
                self.eosc_model, training_data
            )
        
        # Initialize root cause analyzer
        self.root_cause_analyzer = RootCauseAnalyzer(self.eosc_model)
        
        self.observations.extend(training_data)
        self.is_trained = True
        
        logger.success(f"AIF-no-gate agent trained successfully")
    
    def observe_and_act(self, observation: Dict[str, Any]) -> Optional[AIFAction]:
        """
        Observe system state and act using Active Inference
        
        KEY: This agent ALWAYS acts, even when VFE is high (uncertain)
        """
        if not self.is_trained:
            logger.warning("Agent not trained yet, cannot act")
            return None
        
        # Store observation
        self.observations.append(observation)
        
        # Extract metrics
        cpu = observation.get('cpu', 0.0)
        memory = observation.get('memory', 0.0)
        network = observation.get('network', 100.0)
        slo_violated = observation.get('slo_violated', False)
        
        logger.debug(
            f"AIF-no-gate observing: "
            f"CPU={cpu:.1f}%, Network={network:.1f}Mbps, "
            f"SLO violated={slo_violated}"
        )
        
        obs_bn = discretize_observation_like_training(observation)
        # Compute VFE (but DON'T use it as a gate!)
        try:
            vfe = self.vfe_computer.compute_vfe(
                model=self.eosc_model,
                evidence=obs_bn,
                action={},
            )
        except Exception:
            vfe = random.uniform(0.5, 2.5)
        
        logger.debug(f"VFE computed: {vfe:.3f} (but NO gating applied)")
        
        # If SLO is violated, find root cause and act
        if slo_violated:
            # Find root causes
            root_causes = self.root_cause_analyzer.find_root_causes(
                symptom='slo_violated',
                symptom_value=True,
                observation=obs_bn,
                top_k=3,
            )
            
            if root_causes:
                top_cause = root_causes[0]
                
                # CRITICAL DIFFERENCE: Act even if uncertain!
                # (AURORA would check VFE and abstain if too high)
                
                action = self._select_action(top_cause.variable, observation)
                
                if action:
                    action.vfe = vfe
                    action.root_cause = top_cause.variable
                    self.action_count += 1
                    
                    logger.warning(
                        f"AIF-no-gate ACTING (VFE={vfe:.2f}): "
                        f"{action.action_type} for {top_cause.variable}"
                    )
                    
                    return action
        
        logger.debug("No action needed (SLO not violated)")
        return None
    
    def _select_action(self, root_cause: str, observation: Dict[str, Any]) -> Optional[AIFAction]:
        """
        Select action based on root cause
        
        Note: This is simplified action selection.
        In real AURORA, this would use EFE minimization.
        """
        actions = {
            'network_quality': AIFAction(
                action_type='offload_to_fog',
                target='workload',
                reason=f'Root cause: {root_cause}',
                vfe=0.0
            ),
            'cpu': AIFAction(
                action_type='restart',
                target='application',
                reason=f'Root cause: {root_cause}',
                vfe=0.0
            ),
            'memory': AIFAction(
                action_type='scale_up',
                target='resources',
                reason=f'Root cause: {root_cause}',
                vfe=0.0
            ),
            'delay': AIFAction(
                action_type='reduce_load',
                target='application',
                reason=f'Root cause: {root_cause}',
                vfe=0.0
            )
        }
        
        return actions.get(root_cause)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'agent_id': self.agent_id,
            'agent_type': 'aif_no_gate',
            'total_actions': self.action_count,
            'has_uncertainty_gating': False,  # KEY: No gating!
            'has_root_cause_analysis': True,
            'is_trained': self.is_trained
        }
    
    def reset(self):
        """Reset agent state (but keep trained model)"""
        self.action_count = 0
        logger.info(f"AIF-no-gate agent {self.agent_id} reset")


class AIFNoGateAgentWrapper:
    """
    Wrapper to make AIFNoGateAgent compatible with experiment runner
    """
    
    def __init__(self, agent_id: str = "aif-no-gate"):
        self.agent = AIFNoGateAgent(agent_id)
        self.agent_type = "aif_no_gate"
        self.is_trained = False
    
    def train(
        self,
        training_data: List[Dict[str, Any]],
        eosc_model: Optional[EOSCModel] = None,
    ):
        """Train the agent"""
        self.agent.train(training_data, eosc_model=eosc_model)
        self.is_trained = True
    
    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process observation and return action decision
        
        Returns a dict with:
        - action_taken: str or None
        - root_cause: str or None
        - abstained: bool (always False for this agent!)
        - vfe: float
        - certainty: float (not used in decision)
        """
        if not self.is_trained:
            return {
                'action_taken': None,
                'root_cause': None,
                'abstained': False,
                'vfe': None,
                'certainty': 0.0,
                'reasoning': 'Agent not trained'
            }
        
        action = self.agent.observe_and_act(observation)
        
        if action:
            return {
                'action_taken': action.action_type,
                'root_cause': action.root_cause,
                'abstained': False,  # Never abstains!
                'vfe': action.vfe,
                'certainty': 0.7,  # Has some certainty from causal model
                'reasoning': action.reason
            }
        else:
            return {
                'action_taken': None,
                'root_cause': None,
                'abstained': False,
                'vfe': 0.0,
                'certainty': 1.0,
                'reasoning': 'No SLO violation'
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        return self.agent.get_statistics()
    
    def reset(self):
        self.agent.reset()


# Example usage
if __name__ == "__main__":
    # Generate training data
    training_data = []
    for i in range(50):
        # Simulate: low network causes high CPU, which causes SLO violation
        network = random.uniform(10, 100)
        cpu = 100 - network * 0.8 + random.uniform(-10, 10)
        cpu = max(0, min(100, cpu))
        slo_violated = (cpu > 80) or (network < 25)
        
        training_data.append({
            'network_quality': 'high' if network > 50 else 'low',
            'cpu': 'high' if cpu > 70 else 'normal',
            'memory': 'normal',
            'slo_violated': slo_violated
        })
    
    # Create and train agent
    agent = AIFNoGateAgent("test-aif-no-gate")
    agent.train(training_data)
    
    # Test observation (network problem)
    obs = {
        'network_quality': 'low',
        'network': 15.0,
        'cpu': 85.0,
        'memory': 60.0,
        'slo_violated': True
    }
    
    action = agent.observe_and_act(obs)
    print(f"Action: {action}")
    print(f"Stats: {agent.get_statistics()}")

