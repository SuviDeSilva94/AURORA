#!/usr/bin/env python3
"""
Computing continuum demo: edge (CCTV Pi pipeline) + explicit fog handoff + FogNode cluster.

This addresses thesis gaps in one runnable script (still single machine, but *structured* as:
  - Edge: MultiAgentCoordinator on synthetic congestion (same as real-world scenario)
  - Fog API: healing tools call FogTierReceiver (stand-in for HTTP to fog)
  - Cluster: Sedlak-style FogNode + two edge Agent registrations (short background loop)

Run: python3 demo_03_continuum_fog_handoff.py
Strict (85% certainty, may abstain): STRICT=1 python3 demo_03_continuum_fog_handoff.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from loguru import logger
except ModuleNotFoundError as exc:
    print(
        "\nMissing dependencies. Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from src.aif.bayesian_network import BayesianNetworkLearner
from src.aif.certainty_check_agent import CertaintyCheckAgent
from src.aif.continuum_bridge import FogTierReceiver, register_fog_offload_tools
from src.aif.fognode import FogNode
from src.aif.healing_tools import reset_healing_tool_registry
from src.aif.multi_agent_system import (
    DetectorAgent,
    ExecutorAgent,
    MultiAgentCoordinator,
    RootCauseAgent,
    SolutionPlannerAgent,
)
from src.aif.agent import Agent
from src.utils.config import AgentConfig, FogNodeConfig
from demo_02_cctv_edge_scenario import generate_cctv_causal_training_data


def main() -> None:
    strict = os.environ.get("STRICT") == "1"
    certainty_threshold = 0.85 if strict else 0.55
    if not strict:
        logger.info("Using certainty_threshold={} so mitigation + fog handoff usually run; use STRICT=1 for 85%.", certainty_threshold)

    logger.info("=" * 72)
    logger.info("COMPUTING CONTINUUM DEMO — Edge (RCA + healing) → Fog handoff → FogNode cluster")
    logger.info("=" * 72)

    # Fresh registry so this demo always registers fog tools
    reset_healing_tool_registry()
    fog_receiver = FogTierReceiver(node_id="fog-stockholm-001")
    register_fog_offload_tools(fog_receiver)

    training_data = generate_cctv_causal_training_data(150)
    bn_learner = BayesianNetworkLearner()
    eosc_model = bn_learner.learn_structure(training_data)
    eosc_model = bn_learner.update_parameters(eosc_model, training_data)

    detector = DetectorAgent(
        "detector-001",
        thresholds={
            "fps_bad": 0.5,
            "delay_bad": 0.5,
            "throughput_bad": 0.5,
            "slo_violated": 0.5,
        },
    )
    rca_agents = [
        RootCauseAgent("rca-network", eosc_model),
        RootCauseAgent("rca-performance", eosc_model),
    ]
    planner = SolutionPlannerAgent("planner-001", eosc_model)
    executor = ExecutorAgent("executor-001", eosc_model, vfe_threshold=2.0)
    certainty_agent = CertaintyCheckAgent(
        agent_id="certainty-001",
        eosc_model=eosc_model,
        certainty_threshold=certainty_threshold,
        vfe_threshold=2.0,
    )
    coordinator = MultiAgentCoordinator(
        device_id="raspberry-pi-edge-001",
        certainty_threshold=certainty_threshold,
    )
    coordinator.register_agents(
        detector, rca_agents, planner, executor, certainty_check_agent=certainty_agent
    )

    congestion = {
        "fps": 18,
        "delay": 80,
        "throughput": 0.9,
        "cpu": 75,
        "memory": 65,
        "camera_id": "camera_3",
        "node_id": "raspberry-pi-edge-001",
        "fps_bad": 1,
        "delay_bad": 1,
        "throughput_bad": 1,
        "bandwidth_low": 1,
        "node_overload": 0,
        "slo_violated": 1,
    }

    logger.info("\n--- TIER: EDGE — MultiAgentCoordinator.process_observation ---")
    result = coordinator.process_observation(congestion)

    logger.info("\n--- EDGE RESULT ---")
    logger.info("root_cause={} action_taken={} abstained={}", result.get("root_cause"), result.get("action_taken"), result.get("abstained"))
    logger.info("\n--- TIER: FOG — payloads received by FogTierReceiver ---")
    if fog_receiver.received:
        for i, p in enumerate(fog_receiver.received, 1):
            logger.success("  [{}] {}", i, p.get("fog_node_id"))
    else:
        logger.warning("  (none — planner may have chosen an action other than offload, or pipeline abstained)")

    logger.info("\n--- TIER: FOG — Sedlak-style FogNode (2 edge agents, short coordination loop) ---")
    cfg = FogNodeConfig(coordination_interval_ms=400, offloading_enabled=True)
    fog_node = FogNode("fog-node-1", "cctv-cluster-1", cfg)
    edge_a = Agent("edge-pi-001", "cctv_edge", AgentConfig())
    edge_a.slo_fulfillment_rate = 0.45
    edge_a.current_parameters["streams"] = 2
    edge_b = Agent("edge-pi-002", "cctv_edge", AgentConfig())
    edge_b.slo_fulfillment_rate = 0.92
    edge_b.current_parameters["streams"] = 0
    fog_node.register_device(edge_a)
    fog_node.register_device(edge_b)
    fog_node.start()
    time.sleep(1.2)
    fog_node.stop()
    logger.info("FogNode cluster status: {}", fog_node.get_cluster_status())

    logger.success("\nContinuum demo complete. See SCOPE_AND_NEXT_STEPS.md for real deployment steps.")


if __name__ == "__main__":
    main()
