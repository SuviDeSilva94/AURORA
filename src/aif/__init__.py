"""Active Inference Framework"""

# Use absolute imports to avoid circular import issues
from src.aif.agent import Agent
from src.aif.fognode import FogNode
from src.aif.bayesian_network import EOSCModel, BayesianNetworkLearner
from src.aif.vfe_compute import VFEComputer, UncertaintyGate

# Certainty check agent (dedicated micro-agent: "is the decision certain or uncertain?")
try:
    from src.aif.certainty_check_agent import CertaintyCheckAgent, CertaintyCheckResult
except ImportError:
    CertaintyCheckAgent = None
    CertaintyCheckResult = None

# Functional tool calling for self-healing
try:
    from src.aif.healing_tools import HealingToolRegistry, get_healing_tool_registry, ToolResult
except ImportError:
    HealingToolRegistry = None
    get_healing_tool_registry = None
    ToolResult = None

# CCTV specialized agents
try:
    from src.aif.specialized_agents import (
        BandwidthCheckAgent,
        CameraTrafficAgent,
        NodeLoadAgent,
        NetworkDelayAgent,
        ParallelCheckCoordinator,
        AgentFinding
    )
except ImportError:
    # Fallback if not available
    BandwidthCheckAgent = None
    CameraTrafficAgent = None
    NodeLoadAgent = None
    NetworkDelayAgent = None
    ParallelCheckCoordinator = None
    AgentFinding = None

__all__ = [
    'Agent',
    'FogNode',
    'EOSCModel',
    'BayesianNetworkLearner',
    'VFEComputer',
    'UncertaintyGate',
    'CertaintyCheckAgent',
    'CertaintyCheckResult',
    'HealingToolRegistry',
    'get_healing_tool_registry',
    'ToolResult',
    'BandwidthCheckAgent',
    'CameraTrafficAgent',
    'NodeLoadAgent',
    'NetworkDelayAgent',
    'ParallelCheckCoordinator',
    'AgentFinding'
]

