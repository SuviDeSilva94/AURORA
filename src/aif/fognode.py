"""
Fog Node for Cluster-Level Orchestration

Based on Sedlak et al. (2024) - manages device clusters and performs load offloading
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger

from src.aif.agent import Agent
from src.aif.bayesian_network import EOSCModel, BayesianNetworkLearner
from src.aif.vfe_compute import VFEComputer
from src.utils.config import FogNodeConfig


@dataclass
class DeviceCluster:
    """Represents a cluster of edge devices"""
    cluster_id: str
    devices: List[Agent]
    leader_node: 'FogNode'


class FogNode:
    """
    Fog Node for Cluster-Level Coordination
    
    Responsibilities:
    1. Aggregate SLO compliance across device cluster
    2. Facilitate model transfer between heterogeneous devices
    3. Perform load offloading to recover failing devices
    4. Maintain cluster-level EOSC-F model
    
    The fog node ensures the cluster-wide SLO: slo_rate = max(slo_rate)
    """
    
    def __init__(
        self,
        node_id: str,
        cluster_id: str,
        config: FogNodeConfig
    ):
        self.node_id = node_id
        self.cluster_id = cluster_id
        self.config = config
        
        # Cluster management
        self.devices: Dict[str, Agent] = {}  # device_id -> Agent
        self.device_models: Dict[str, EOSCModel] = {}  # device_type -> model
        
        # Cluster-level AIF
        self.bn_learner = BayesianNetworkLearner(
            algorithm=config.bn_learning_algorithm
        )
        self.cluster_model: Optional[EOSCModel] = None
        self.vfe_computer = VFEComputer()
        
        # State
        self.running = False
        self.coordination_thread: Optional[threading.Thread] = None
        self.iteration_count = 0
        
        # Metrics
        self.cluster_slo_rate = 0.0
        
        logger.info(f"FogNode {node_id} initialized for cluster {cluster_id}")
    
    def start(self):
        """Start the fog node coordination loop"""
        if self.running:
            logger.warning(f"FogNode {self.node_id} already running")
            return
        
        self.running = True
        self.coordination_thread = threading.Thread(
            target=self._coordination_loop,
            daemon=True
        )
        self.coordination_thread.start()
        logger.info(f"FogNode {self.node_id} started")
    
    def stop(self):
        """Stop the fog node"""
        self.running = False
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5.0)
        logger.info(f"FogNode {self.node_id} stopped")
    
    def register_device(self, device: Agent):
        """Register an edge device with this fog node"""
        self.devices[device.agent_id] = device
        logger.info(
            f"FogNode {self.node_id} registered device {device.agent_id} "
            f"(type={device.device_type})"
        )
    
    def unregister_device(self, device_id: str):
        """Unregister a device from the cluster"""
        if device_id in self.devices:
            del self.devices[device_id]
            logger.info(f"FogNode {self.node_id} unregistered device {device_id}")
    
    def _coordination_loop(self):
        """Main coordination loop for cluster management"""
        while self.running:
            try:
                start_time = time.time()
                
                # 1. Aggregate cluster metrics
                cluster_metrics = self._aggregate_cluster_metrics()
                
                # 2. Evaluate cluster-wide SLO
                self._evaluate_cluster_slo(cluster_metrics)
                
                # 3. Facilitate knowledge transfer
                if self.iteration_count % self.config.knowledge_transfer_interval == 0:
                    self._facilitate_knowledge_transfer()
                
                # 4. Perform load offloading if needed
                if self.config.offloading_enabled:
                    self._perform_load_offloading(cluster_metrics)
                
                # 5. Update cluster-level EOSC-F model
                if self.config.cluster_model_enabled:
                    self._update_cluster_model(cluster_metrics)
                
                self.iteration_count += 1
                
                # Sleep to maintain iteration rate
                elapsed = time.time() - start_time
                sleep_time = max(
                    0,
                    (self.config.coordination_interval_ms / 1000.0) - elapsed
                )
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(
                    f"Error in coordination loop for FogNode {self.node_id}: {e}",
                    exc_info=True
                )
                time.sleep(1.0)
    
    def _aggregate_cluster_metrics(self) -> Dict[str, Any]:
        """
        Aggregate metrics from all devices in the cluster
        
        Returns:
            Cluster-level metrics
        """
        device_statuses = [device.get_status() for device in self.devices.values()]
        
        # Calculate aggregate metrics
        total_devices = len(device_statuses)
        if total_devices == 0:
            return {}
        
        avg_slo_rate = sum(
            d['slo_fulfillment_rate'] for d in device_statuses
        ) / total_devices
        
        avg_vfe = sum(
            d['last_vfe'] for d in device_statuses if d['last_vfe'] != float('inf')
        ) / max(1, sum(1 for d in device_statuses if d['last_vfe'] != float('inf')))
        
        total_abstentions = sum(d['abstention_count'] for d in device_statuses)
        
        return {
            'total_devices': total_devices,
            'avg_slo_rate': avg_slo_rate,
            'avg_vfe': avg_vfe,
            'total_abstentions': total_abstentions,
            'device_statuses': device_statuses
        }
    
    def _evaluate_cluster_slo(self, cluster_metrics: Dict[str, Any]):
        """
        Evaluate cluster-wide SLO: slo_rate = max(slo_rate)
        
        From Table 2: Fog-level SLO ensuring both QoS and QoE
        """
        if not cluster_metrics:
            self.cluster_slo_rate = 0.0
            return
        
        # Goal: maximize average SLO fulfillment rate
        self.cluster_slo_rate = cluster_metrics.get('avg_slo_rate', 0.0)
        
        logger.debug(
            f"FogNode {self.node_id} cluster SLO rate: {self.cluster_slo_rate:.2%}"
        )
    
    def _facilitate_knowledge_transfer(self):
        """
        Facilitate knowledge transfer between devices
        
        Addresses research questions K-1, K-2, K-3:
        - K-1: SLO fulfillment rate of transferred models
        - K-2: Speedup through knowledge transfer
        - K-3: Merged models decrease FE
        
        Strategy:
        1. Group devices by type
        2. For each type, select best-performing model
        3. Transfer to devices that lack models or are underperforming
        """
        logger.info(f"FogNode {self.node_id} facilitating knowledge transfer")
        
        # Group devices by type
        devices_by_type = defaultdict(list)
        for device in self.devices.values():
            devices_by_type[device.device_type].append(device)
        
        # For each device type
        for device_type, devices in devices_by_type.items():
            # Find best-performing device with a model
            best_device = None
            best_slo_rate = 0.0
            
            for device in devices:
                if device.eosc_model is not None:
                    if device.slo_fulfillment_rate > best_slo_rate:
                        best_slo_rate = device.slo_fulfillment_rate
                        best_device = device
            
            if best_device is None:
                continue
            
            # Transfer model to devices without models or low SLO rates
            best_model = best_device.get_model()
            threshold = self.config.knowledge_transfer_threshold
            
            for device in devices:
                if device.agent_id == best_device.agent_id:
                    continue
                
                # Transfer if device has no model or is underperforming
                if (device.eosc_model is None or 
                    device.slo_fulfillment_rate < threshold):
                    device.transfer_model(best_model)
                    logger.info(
                        f"Transferred model from {best_device.agent_id} to "
                        f"{device.agent_id} (type={device_type})"
                    )
            
            # Store the best model for this device type
            self.device_models[device_type] = best_model
    
    def _perform_load_offloading(self, cluster_metrics: Dict[str, Any]):
        """
        Perform load offloading within the cluster
        
        When a device fails to meet SLOs (e.g., due to network issues or
        hardware limitations), redistribute load to other devices.
        
        From paper: "rebalancing the load within a device cluster allowed
        individual edge devices to recover their SLO compliance after a
        network failure from 22% to 89%"
        
        Strategy:
        1. Identify devices with low SLO fulfillment
        2. Identify devices with capacity
        3. Redistribute streams from failing to capable devices
        """
        device_statuses = cluster_metrics.get('device_statuses', [])
        
        # Find devices below threshold
        failing_devices = []
        capable_devices = []
        
        threshold = self.config.offloading_threshold
        
        for status in device_statuses:
            device_id = status['agent_id']
            device = self.devices.get(device_id)
            
            if device is None:
                continue
            
            slo_rate = status['slo_fulfillment_rate']
            
            if slo_rate < threshold:
                failing_devices.append((device, slo_rate))
            elif slo_rate > 0.8:  # Has capacity
                capable_devices.append((device, slo_rate))
        
        if not failing_devices or not capable_devices:
            return
        
        logger.info(
            f"FogNode {self.node_id} performing load offloading: "
            f"{len(failing_devices)} failing, {len(capable_devices)} capable"
        )
        
        # Redistribute load
        for failing_device, _ in failing_devices:
            # Reduce load on failing device
            current_streams = failing_device.current_parameters.get('streams', 1)
            if current_streams > 0:
                new_streams = max(0, current_streams - 1)
                failing_device.current_parameters['streams'] = new_streams
                
                logger.info(
                    f"Reduced streams on {failing_device.agent_id}: "
                    f"{current_streams} -> {new_streams}"
                )
                
                # Assign to capable device
                if capable_devices:
                    capable_device, _ = capable_devices[0]
                    cap_streams = capable_device.current_parameters.get('streams', 0)
                    capable_device.current_parameters['streams'] = cap_streams + 1
                    
                    logger.info(
                        f"Increased streams on {capable_device.agent_id}: "
                        f"{cap_streams} -> {cap_streams + 1}"
                    )
    
    def _update_cluster_model(self, cluster_metrics: Dict[str, Any]):
        """
        Update cluster-level EOSC-F model
        
        The fog node maintains its own causal model of cluster dynamics
        Variables: avg_slo_rate, total_devices, congestion, etc.
        """
        # Placeholder for cluster-level learning
        # Would use similar BN learning as Agent, but with cluster-level variables
        pass
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status for monitoring"""
        return {
            'node_id': self.node_id,
            'cluster_id': self.cluster_id,
            'running': self.running,
            'iteration_count': self.iteration_count,
            'cluster_slo_rate': self.cluster_slo_rate,
            'total_devices': len(self.devices),
            'device_types': list(self.device_models.keys())
        }

