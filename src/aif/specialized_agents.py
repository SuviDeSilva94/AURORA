"""
Specialized Micro-Agents for CCTV Monitoring
Each agent performs ONE specific check (  requirement)
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class AgentFinding:
    """Result from a specialized agent check"""
    agent_id: str
    issue_detected: bool
    issue_type: Optional[str]
    metric_values: Dict[str, Any]
    confidence: float
    recommendation: Optional[str]


class BandwidthCheckAgent:
    """
    Micro-agent that ONLY checks network bandwidth
    Runs on each edge node
    """
    def __init__(self, agent_id: str, threshold_mbps: float = 1.6):
        self.agent_id = agent_id
        self.threshold = threshold_mbps
        logger.info(f"[{agent_id}] Bandwidth check agent initialized (threshold={threshold_mbps} MB/s)")
    
    def check(self, observation: Dict[str, Any]) -> AgentFinding:
        """Check if bandwidth is sufficient"""
        throughput = observation.get('throughput', 0)
        
        if throughput < self.threshold:
            return AgentFinding(
                agent_id=self.agent_id,
                issue_detected=True,
                issue_type='bandwidth_drop',
                metric_values={'throughput': throughput, 'threshold': self.threshold},
                confidence=min(1.0, (self.threshold - throughput) / self.threshold),
                recommendation='offload_to_fog'
            )
        
        return AgentFinding(
            agent_id=self.agent_id,
            issue_detected=False,
            issue_type=None,
            metric_values={'throughput': throughput},
            confidence=1.0,
            recommendation=None
        )


class CameraTrafficAgent:
    """
    Micro-agent that ONLY checks per-camera traffic patterns
    Detects cameras sending excessive frames
    """
    def __init__(self, agent_id: str, fps_threshold: int = 35):
        self.agent_id = agent_id
        self.fps_threshold = fps_threshold
        logger.info(f"[{agent_id}] Camera traffic agent initialized (fps_threshold={fps_threshold})")
    
    def check(self, observation: Dict[str, Any]) -> AgentFinding:
        """Check for excessive frame transmission"""
        fps = observation.get('fps', 30)
        camera_id = observation.get('camera_id', 'unknown')
        
        if fps > self.fps_threshold:
            return AgentFinding(
                agent_id=self.agent_id,
                issue_detected=True,
                issue_type='excessive_frames',
                metric_values={'fps': fps, 'camera_id': camera_id, 'threshold': self.fps_threshold},
                confidence=min(1.0, (fps - self.fps_threshold) / self.fps_threshold),
                recommendation='rate_limit_camera'
            )
        
        return AgentFinding(
            agent_id=self.agent_id,
            issue_detected=False,
            issue_type=None,
            metric_values={'fps': fps, 'camera_id': camera_id},
            confidence=1.0,
            recommendation=None
        )


class NodeLoadAgent:
    """
    Micro-agent that ONLY checks edge node (Raspberry Pi) CPU/memory
    Detects node overload conditions
    """
    def __init__(self, agent_id: str, cpu_threshold: float = 85, memory_threshold: float = 85):
        self.agent_id = agent_id
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        logger.info(f"[{agent_id}] Node load agent initialized (cpu={cpu_threshold}%, mem={memory_threshold}%)")
    
    def check(self, observation: Dict[str, Any]) -> AgentFinding:
        """Check for node overload"""
        cpu = observation.get('cpu', 0)
        memory = observation.get('memory', 0)
        node_id = observation.get('node_id', 'unknown')
        
        if cpu > self.cpu_threshold or memory > self.memory_threshold:
            return AgentFinding(
                agent_id=self.agent_id,
                issue_detected=True,
                issue_type='node_overload',
                metric_values={
                    'cpu': cpu, 
                    'memory': memory, 
                    'node_id': node_id,
                    'cpu_threshold': self.cpu_threshold,
                    'memory_threshold': self.memory_threshold
                },
                confidence=max(
                    (cpu - self.cpu_threshold) / self.cpu_threshold if cpu > self.cpu_threshold else 0,
                    (memory - self.memory_threshold) / self.memory_threshold if memory > self.memory_threshold else 0
                ),
                recommendation='redistribute_load'
            )
        
        return AgentFinding(
            agent_id=self.agent_id,
            issue_detected=False,
            issue_type=None,
            metric_values={'cpu': cpu, 'memory': memory, 'node_id': node_id},
            confidence=1.0,
            recommendation=None
        )


class NetworkDelayAgent:
    """
    Micro-agent that ONLY checks network delay/latency
    Detects delay spikes that violate CCTV SLOs
    """
    def __init__(self, agent_id: str, delay_threshold_ms: float = 33):
        self.agent_id = agent_id
        self.delay_threshold = delay_threshold_ms
        logger.info(f"[{agent_id}] Network delay agent initialized (threshold={delay_threshold_ms}ms)")
    
    def check(self, observation: Dict[str, Any]) -> AgentFinding:
        """Check for delay spikes"""
        delay = observation.get('delay', 0)
        
        if delay > self.delay_threshold:
            return AgentFinding(
                agent_id=self.agent_id,
                issue_detected=True,
                issue_type='delay_spike',
                metric_values={'delay': delay, 'threshold': self.delay_threshold},
                confidence=min(1.0, (delay - self.delay_threshold) / self.delay_threshold),
                recommendation='offload_to_fog'
            )
        
        return AgentFinding(
            agent_id=self.agent_id,
            issue_detected=False,
            issue_type=None,
            metric_values={'delay': delay},
            confidence=1.0,
            recommendation=None
        )


class ParallelCheckCoordinator:
    """
    Coordinates parallel execution of specialized micro-agents
    Implements  's requirement for parallel small agents
    """
    def __init__(self):
        self.agents = [
            BandwidthCheckAgent("bandwidth-check-001"),
            CameraTrafficAgent("camera-traffic-001"),
            NodeLoadAgent("node-load-001"),
            NetworkDelayAgent("network-delay-001")
        ]
        logger.info(f"Parallel check coordinator initialized with {len(self.agents)} agents")
    
    def parallel_check(self, observation: Dict[str, Any]) -> List[AgentFinding]:
        """
        Run all specialized agents in parallel
        Returns list of findings from all agents
        """
        logger.info("=" * 60)
        logger.info("PARALLEL AGENT CHECKS (  Requirement)")
        logger.info("=" * 60)
        
        findings = []
        
        # In production, these would run in parallel threads/processes
        # For now, sequential execution (same output/function)
        for agent in self.agents:
            finding = agent.check(observation)
            findings.append(finding)
            
            if finding.issue_detected:
                logger.warning(
                    f"[{finding.agent_id}] ISSUE DETECTED: {finding.issue_type} "
                    f"(confidence={finding.confidence:.2%})"
                )
            else:
                logger.debug(f"[{finding.agent_id}] No issues detected")
        
        # Filter to only detected issues
        detected_issues = [f for f in findings if f.issue_detected]
        
        logger.info(f"Parallel checks complete: {len(detected_issues)}/{len(findings)} agents detected issues")
        
        return findings

