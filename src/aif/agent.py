"""
Active Inference Agent for Edge Devices

Based on Sedlak et al. (2024) "Equilibrium in the Computing Continuum through Active Inference"
https://github.com/borissedlak/workload/tree/main/FGCS

This agent uses Bayesian Networks to develop a causal understanding of how to enforce SLOs.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from loguru import logger

from src.aif.bayesian_network import BayesianNetworkLearner, EOSCModel
from src.aif.vfe_compute import VFEComputer, UncertaintyGate
from src.utils.config import AgentConfig


@dataclass
class Observation:
    """Observation collected during AIF iteration"""
    timestamp: float
    metrics: Dict[str, Any]
    slo_violations: List[str] = field(default_factory=list)


@dataclass
class Action:
    """Action to be executed by the agent"""
    parameter: str  # e.g., 'pixel', 'fps', 'streams'
    value: Any
    vfe: float
    timestamp: float


class Agent:
    """
    Active Inference Agent for Edge Devices
    
    Implements the core AIF cycle:
    1. Observe: Collect metrics and SLO states
    2. Infer: Use EOSC model to predict outcomes
    3. Act: Select action that minimizes expected free energy
    4. Learn: Update Bayesian Network with new observations
    
    Uses VFE threshold: only acts when confident (after root cause is identified with high certainty).
    """
    
    def __init__(
        self,
        agent_id: str,
        device_type: str,
        config: AgentConfig,
        fog_node_address: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.device_type = device_type
        self.config = config
        self.fog_node_address = fog_node_address
        
        # Active Inference components
        self.bn_learner = BayesianNetworkLearner(
            algorithm=config.bn_learning_algorithm,
            use_markov_blanket=config.mb_enabled
        )
        self.eosc_model: Optional[EOSCModel] = None
        self.vfe_computer = VFEComputer()
        self.uncertainty_gate = UncertaintyGate(threshold=config.vfe_threshold)
        
        # Data collection
        self.observation_buffer: List[Observation] = []
        self.training_data: List[Dict[str, Any]] = []
        
        # State
        self.current_parameters: Dict[str, Any] = {}
        self.running = False
        self.aif_thread: Optional[threading.Thread] = None
        self.iteration_count = 0
        
        # Metrics
        self.slo_fulfillment_rate = 0.0
        self.last_vfe = float('inf')
        self.abstention_count = 0
        
        logger.info(f"Agent {agent_id} initialized (device_type={device_type})")
    
    def start(self):
        """Start the AIF cycle in a separate thread"""
        if self.running:
            logger.warning(f"Agent {self.agent_id} already running")
            return
        
        self.running = True
        self.aif_thread = threading.Thread(target=self._aif_loop, daemon=True)
        self.aif_thread.start()
        logger.info(f"Agent {self.agent_id} started")
    
    def stop(self):
        """Stop the AIF cycle"""
        self.running = False
        if self.aif_thread:
            self.aif_thread.join(timeout=5.0)
        logger.info(f"Agent {self.agent_id} stopped")
    
    def _aif_loop(self):
        """Main Active Inference loop"""
        while self.running:
            try:
                start_time = time.time()
                
                # 1. OBSERVE: Collect metrics and evaluate SLOs
                observation = self._observe()
                self.observation_buffer.append(observation)
                
                # 2. LEARN: Update Bayesian Network structure and parameters
                if self._should_learn():
                    self._learn()
                
                # 3. INFER: Use EOSC model to predict outcomes
                if self.eosc_model is not None:
                    action = self._infer_best_action(observation)
                    
                    # 4. ACT: Execute action if VFE is below threshold
                    if action and self.uncertainty_gate.should_act(action.vfe):
                        self._execute_action(action)
                    else:
                        self.abstention_count += 1
                        logger.warning(
                            f"Agent {self.agent_id} abstaining from action "
                            f"(VFE={action.vfe:.3f} > threshold={self.config.vfe_threshold})"
                        )
                
                self.iteration_count += 1
                
                # Sleep to maintain iteration rate
                elapsed = time.time() - start_time
                sleep_time = max(0, (self.config.aif_iteration_ms / 1000.0) - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in AIF loop for agent {self.agent_id}: {e}", exc_info=True)
                time.sleep(1.0)
    
    def _observe(self) -> Observation:
        """
        Collect current metrics and evaluate SLO violations
        
        Returns:
            Observation with current system state
        """
        # This should be overridden or injected with actual metric collection
        metrics = self._collect_metrics()
        slo_violations = self._evaluate_slos(metrics)
        
        return Observation(
            timestamp=time.time(),
            metrics=metrics,
            slo_violations=slo_violations
        )
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """
        Collect device and application metrics
        
        Edge metrics (from Table 1):
        - pixel, fps, bitrate (from IoT)
        - cpu, memory, streams, consumption, network (from Edge)
        - delay, success, distance (from Application)
        - slo_rate, device_type, congestion
        """
        # Placeholder - should be implemented based on actual monitoring
        return {
            'pixel': 640 * 480,  # Resolution
            'fps': 30,
            'bitrate': 0.0,
            'cpu': 0.0,
            'memory': 0.0,
            'streams': 1,
            'consumption': 0.0,
            'network': 0.0,
            'delay': 0.0,
            'success': True,
            'distance': 0,
            'slo_rate': 0.0,
            'device_type': self.device_type,
            'congestion': 0
        }
    
    def _evaluate_slos(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Evaluate SLO compliance based on Table 2 from Sedlak et al.
        
        SLOs:
        - network: throughput < 1.6 MB/s
        - in_time: delay < 1/fps
        - success: success == True
        - distance: distance < 50
        """
        violations = []
        
        # Network SLO (QoS)
        if metrics.get('network', 0) >= 1.6:
            violations.append('network')
        
        # In-time SLO (QoS)
        fps = metrics.get('fps', 30)
        max_delay = (1.0 / fps) * 1000  # Convert to ms
        if metrics.get('delay', 0) >= max_delay:
            violations.append('in_time')
        
        # Success SLO (QoE)
        if not metrics.get('success', True):
            violations.append('success')
        
        # Distance SLO (QoE)
        if metrics.get('distance', 0) >= 50:
            violations.append('distance')
        
        # Update SLO fulfillment rate
        total_slos = 4
        fulfilled = total_slos - len(violations)
        self.slo_fulfillment_rate = fulfilled / total_slos
        
        return violations
    
    def _should_learn(self) -> bool:
        """Determine if BN learning should be triggered"""
        # Learn every N iterations or when buffer is full
        batch_size = self.config.learning_batch_size
        return len(self.observation_buffer) >= batch_size
    
    def _learn(self):
        """
        Update EOSC model using Bayesian Network Learning
        
        Steps:
        1. Convert observations to training data
        2. Structure Learning (STRL): Learn DAG structure
        3. Parameter Learning (PARL): Update CPTs
        4. Extract Markov Blanket if enabled
        """
        logger.info(f"Agent {self.agent_id} learning from {len(self.observation_buffer)} observations")
        
        # Convert observations to DataFrame
        data = [obs.metrics for obs in self.observation_buffer]
        self.training_data.extend(data)
        
        # Learn or update EOSC model
        if self.eosc_model is None:
            # Initial structure learning
            self.eosc_model = self.bn_learner.learn_structure(self.training_data)
            logger.info(f"Agent {self.agent_id} learned initial EOSC structure")
        else:
            # Update existing model (PARL)
            self.eosc_model = self.bn_learner.update_parameters(
                self.eosc_model,
                data
            )
            logger.info(f"Agent {self.agent_id} updated EOSC parameters")
        
        # Clear observation buffer
        self.observation_buffer.clear()
    
    def _infer_best_action(self, observation: Observation) -> Optional[Action]:
        """
        Use EOSC model to infer the best action
        
        For each parameterizable variable:
        1. Compute Expected Free Energy (EFE) for possible values
        2. Select action with minimum EFE
        3. Check VFE uncertainty threshold
        
        Returns:
            Action with minimum VFE, or None if no valid action
        """
        if self.eosc_model is None:
            return None
        
        # Parameterizable variables (from Table 1)
        param_variables = ['pixel', 'fps']  # Edge-level parameters
        
        best_action = None
        min_vfe = float('inf')
        
        for param in param_variables:
            # Get possible values for this parameter
            possible_values = self._get_possible_values(param)
            
            for value in possible_values:
                # Compute VFE for this action
                vfe = self.vfe_computer.compute_vfe(
                    model=self.eosc_model,
                    evidence=observation.metrics,
                    action={param: value}
                )
                
                if vfe < min_vfe:
                    min_vfe = vfe
                    best_action = Action(
                        parameter=param,
                        value=value,
                        vfe=vfe,
                        timestamp=time.time()
                    )
        
        self.last_vfe = min_vfe
        return best_action
    
    def _get_possible_values(self, parameter: str) -> List[Any]:
        """Get possible values for a parameter"""
        # Simplified - should be based on device capabilities
        if parameter == 'pixel':
            return [320*240, 640*480, 1280*720, 1920*1080]
        elif parameter == 'fps':
            return [10, 15, 20, 30]
        else:
            return []
    
    def _execute_action(self, action: Action):
        """
        Execute the selected action
        
        Args:
            action: Action to execute
        """
        logger.info(
            f"Agent {self.agent_id} executing action: "
            f"{action.parameter}={action.value} (VFE={action.vfe:.3f})"
        )
        
        # Update current parameters
        self.current_parameters[action.parameter] = action.value
        
        # Actual execution would reconfigure the device/application
        # For now, just store the parameter change
    
    def transfer_model(self, model: EOSCModel):
        """
        Receive a transferred EOSC model from another device
        
        Enables knowledge transfer (K-1, K-2 from paper)
        """
        logger.info(f"Agent {self.agent_id} received transferred model")
        self.eosc_model = model
    
    def get_model(self) -> Optional[EOSCModel]:
        """Get the current EOSC model for transfer"""
        return self.eosc_model
    
    def merge_model(self, other_model: EOSCModel):
        """
        Merge another model with current model
        
        Addresses K-3: Can merged models decrease FE?
        """
        if self.eosc_model is None:
            self.eosc_model = other_model
        else:
            self.eosc_model = self.bn_learner.merge_models(
                self.eosc_model,
                other_model
            )
            logger.info(f"Agent {self.agent_id} merged models")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status for monitoring"""
        return {
            'agent_id': self.agent_id,
            'device_type': self.device_type,
            'running': self.running,
            'iteration_count': self.iteration_count,
            'slo_fulfillment_rate': self.slo_fulfillment_rate,
            'last_vfe': self.last_vfe,
            'abstention_count': self.abstention_count,
            'has_model': self.eosc_model is not None,
            'current_parameters': self.current_parameters
        }

