"""
Rule-Based Baseline Agent

A simple threshold-based agent for comparison with AURORA.
This represents traditional monitoring/alerting approaches.

Logic:
- If CPU > 80% → restart application
- If memory > 85% → scale up resources
- If network < 20 Mbps → offload to fog
- If delay > threshold → reduce load

No uncertainty quantification, no root cause analysis.
Just simple if-then rules.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class RuleAction:
    """An action suggested by a rule"""
    action_type: str
    target: str
    reason: str
    confidence: float = 1.0  # Rules always have 100% confidence


class RuleBasedAgent:
    """
    Simple threshold-based agent (baseline for comparison)
    
    This agent represents traditional monitoring approaches:
    - No causal reasoning
    - No uncertainty quantification
    - No root cause analysis
    - Just threshold-based rules
    """
    
    def __init__(self, agent_id: str = "rule-based"):
        self.agent_id = agent_id
        
        # Thresholds (tunable parameters)
        self.cpu_high_threshold = 80.0
        self.memory_high_threshold = 85.0
        self.network_low_threshold = 20.0  # Mbps
        self.delay_high_threshold = 100.0  # ms
        self.fps_low_threshold = 20.0
        
        self.action_count = 0
        logger.info(f"Rule-based agent {agent_id} initialized")
    
    def observe_and_act(self, observation: Dict[str, Any]) -> Optional[RuleAction]:
        """
        Observe system state and apply simple threshold rules
        
        Returns:
            RuleAction if a rule fires, None otherwise
        """
        # Extract metrics
        cpu = observation.get('cpu', 0.0)
        memory = observation.get('memory', 0.0)
        network = observation.get('network', 100.0)
        delay = observation.get('delay', 0.0)
        fps = observation.get('fps', 30.0)
        slo_violated = observation.get('slo_violated', False)
        
        logger.debug(
            f"Rule-based agent observing: "
            f"CPU={cpu:.1f}%, Memory={memory:.1f}%, "
            f"Network={network:.1f}Mbps, Delay={delay:.1f}ms"
        )
        
        # Rule 1: High CPU → Restart application
        if cpu > self.cpu_high_threshold:
            action = RuleAction(
                action_type="restart",
                target="application",
                reason=f"CPU usage ({cpu:.1f}%) exceeds threshold ({self.cpu_high_threshold}%)"
            )
            self.action_count += 1
            logger.warning(f"RULE FIRED: {action.reason} → {action.action_type}")
            return action
        
        # Rule 2: High memory → Scale up
        if memory > self.memory_high_threshold:
            action = RuleAction(
                action_type="scale_up",
                target="resources",
                reason=f"Memory usage ({memory:.1f}%) exceeds threshold ({self.memory_high_threshold}%)"
            )
            self.action_count += 1
            logger.warning(f"RULE FIRED: {action.reason} → {action.action_type}")
            return action
        
        # Rule 3: Low network → Offload to fog
        if network < self.network_low_threshold:
            action = RuleAction(
                action_type="offload_to_fog",
                target="workload",
                reason=f"Network bandwidth ({network:.1f}Mbps) below threshold ({self.network_low_threshold}Mbps)"
            )
            self.action_count += 1
            logger.warning(f"RULE FIRED: {action.reason} → {action.action_type}")
            return action
        
        # Rule 4: High delay → Reduce load
        if delay > self.delay_high_threshold:
            action = RuleAction(
                action_type="reduce_load",
                target="application",
                reason=f"Processing delay ({delay:.1f}ms) exceeds threshold ({self.delay_high_threshold}ms)"
            )
            self.action_count += 1
            logger.warning(f"RULE FIRED: {action.reason} → {action.action_type}")
            return action
        
        # Rule 5: Low FPS → Reduce quality
        if fps < self.fps_low_threshold and slo_violated:
            action = RuleAction(
                action_type="reduce_quality",
                target="video_stream",
                reason=f"FPS ({fps:.1f}) below threshold ({self.fps_low_threshold})"
            )
            self.action_count += 1
            logger.warning(f"RULE FIRED: {action.reason} → {action.action_type}")
            return action
        
        # No rule fired
        logger.debug("No rule triggered, system appears normal")
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'agent_id': self.agent_id,
            'agent_type': 'rule_based',
            'total_actions': self.action_count,
            'has_uncertainty_gating': False,
            'has_root_cause_analysis': False
        }
    
    def reset(self):
        """Reset agent state"""
        self.action_count = 0
        logger.info(f"Rule-based agent {self.agent_id} reset")


class RuleBasedAgentWrapper:
    """
    Wrapper to make RuleBasedAgent compatible with experiment runner
    """
    
    def __init__(self, agent_id: str = "rule-based"):
        self.agent = RuleBasedAgent(agent_id)
        self.agent_type = "rule_based"
    
    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process observation and return action decision
        
        Returns a dict with:
        - action_taken: str or None
        - root_cause: str or None
        - abstained: bool
        - vfe: None (rule-based doesn't compute VFE)
        - certainty: 1.0 (rules always certain)
        """
        action = self.agent.observe_and_act(observation)
        
        if action:
            return {
                'action_taken': action.action_type,
                'root_cause': 'symptom_based',  # Rules don't identify root causes
                'abstained': False,
                'vfe': None,
                'certainty': action.confidence,
                'reasoning': action.reason
            }
        else:
            return {
                'action_taken': None,
                'root_cause': None,
                'abstained': False,
                'vfe': None,
                'certainty': 1.0,
                'reasoning': 'No threshold exceeded'
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        return self.agent.get_statistics()
    
    def reset(self):
        self.agent.reset()


# Example usage
if __name__ == "__main__":
    agent = RuleBasedAgent("baseline-1")
    
    # Normal observation
    obs1 = {
        'cpu': 45.0,
        'memory': 60.0,
        'network': 50.0,
        'delay': 30.0,
        'fps': 30.0,
        'slo_violated': False
    }
    
    action = agent.observe_and_act(obs1)
    print(f"Normal state: {action}")  # Should be None
    
    # High CPU observation
    obs2 = {
        'cpu': 92.0,
        'memory': 60.0,
        'network': 50.0,
        'delay': 30.0,
        'fps': 30.0,
        'slo_violated': True
    }
    
    action = agent.observe_and_act(obs2)
    print(f"High CPU: {action}")  # Should trigger restart
    
    # Low network observation
    obs3 = {
        'cpu': 45.0,
        'memory': 60.0,
        'network': 15.0,
        'delay': 30.0,
        'fps': 30.0,
        'slo_violated': True
    }
    
    action = agent.observe_and_act(obs3)
    print(f"Low network: {action}")  # Should trigger offload


