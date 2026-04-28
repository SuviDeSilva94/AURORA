#!/usr/bin/env python3
"""
Real-world scenario test: Raspberry Pi CCTV at the edge with network congestion

Thesis scenario (exact):
- Raspberry Pi 4 processing live CCTV security camera footage
- SLOs: 30 FPS, delay < 33 ms, throughput > 1.6 MB/s, success rate ≥95%
- Fault: Sudden network congestion → FPS drop (30→18), delay spike to 80 ms, throughput drop

Pipeline: congestion detection → root cause with high certainty → mitigation (or abstention).

Optional: set DEMO_MITIGATION=1 to use a lower certainty threshold so mitigation runs
(e.g. offload_to_fog) for demo purposes; default keeps 85% for real-world safety.
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random

try:
    from loguru import logger
except ModuleNotFoundError as exc:
    print(
        "\nMissing dependencies (e.g. loguru). From the project folder run:\n"
        "  python3 -m venv venv\n"
        "  source venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
        "Then: python3 demo_02_cctv_edge_scenario.py\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from src.utils.cctv_config import CCTV_SLOS, CCTV_FAULT_TYPES
from src.aif.bayesian_network import BayesianNetworkLearner
from src.aif.multi_agent_system import (
    MultiAgentCoordinator,
    DetectorAgent,
    RootCauseAgent,
    SolutionPlannerAgent,
    ExecutorAgent,
)
from src.aif.certainty_check_agent import CertaintyCheckAgent


def generate_cctv_causal_training_data(num_samples: int = 150):
    """
    Generate training data with causal structure matching the thesis:
    - bandwidth_low / node_overload cause delay_bad, throughput_bad, fps_bad
    - slo_violated when any of these are bad
    Uses 0/1 so BN and detector are consistent.
    """
    data = []
    for _ in range(num_samples):
        bandwidth_low = random.random() < 0.35
        node_overload = random.random() < 0.25
        # Causal: bandwidth_low -> throughput_bad, delay_bad; node_overload -> fps_bad, delay_bad
        throughput_bad = bandwidth_low or (random.random() < 0.2)
        delay_bad = bandwidth_low or node_overload or (random.random() < 0.15)
        fps_bad = node_overload or (random.random() < 0.2)
        slo_violated = 1 if (throughput_bad or delay_bad or fps_bad) else 0
        data.append({
            'bandwidth_low': 1 if bandwidth_low else 0,
            'node_overload': 1 if node_overload else 0,
            'throughput_bad': 1 if throughput_bad else 0,
            'delay_bad': 1 if delay_bad else 0,
            'fps_bad': 1 if fps_bad else 0,
            'slo_violated': slo_violated,
        })
    return data


def main():
    logger.info("=" * 72)
    logger.info("REAL-WORLD SCENARIO: Raspberry Pi CCTV + Network Congestion")
    logger.info("=" * 72)
    logger.info("")
    logger.info("Setup (thesis exact):")
    logger.info("  • Raspberry Pi 4 at edge processing live CCTV footage")
    logger.info("  • SLOs: {} FPS, delay < {} ms, throughput > {} MB/s, success ≥{}%".format(
        CCTV_SLOS['fps'], CCTV_SLOS['network_delay'], CCTV_SLOS['throughput'], CCTV_SLOS['success_rate'] * 100
    ))
    logger.info("  • Fault: sudden network congestion")
    logger.info("")

    # Healthy baseline (for comparison)
    healthy = {
        'fps': 30,
        'delay': 20,
        'throughput': 2.0,
        'cpu': 55,
        'memory': 60,
        'fps_bad': 0,
        'delay_bad': 0,
        'throughput_bad': 0,
        'bandwidth_low': 0,
        'node_overload': 0,
        'slo_violated': 0,
    }
    logger.info("Baseline (healthy):")
    logger.info("  FPS={}, delay={} ms, throughput={} MB/s → SLOs OK", healthy['fps'], healthy['delay'], healthy['throughput'])
    logger.info("")

    # Congestion observation (thesis: FPS drop, delay 80 ms, throughput drop)
    congestion = {
        'fps': 18,
        'delay': 80,
        'throughput': 0.9,
        'cpu': 75,
        'memory': 65,
        'camera_id': 'camera_3',
        'node_id': 'raspberry-pi-001',
        'fps_bad': 1,
        'delay_bad': 1,
        'throughput_bad': 1,
        'bandwidth_low': 1,
        'node_overload': 0,
        'slo_violated': 1,
    }
    logger.warning("After congestion:")
    logger.warning("  FPS={} (target 30), delay={} ms (target <33), throughput={} MB/s (target >1.6)",
                  congestion['fps'], congestion['delay'], congestion['throughput'])
    logger.warning("  SLO violated → root cause must be identified with high certainty before mitigation")
    logger.info("")

    # 1. Train causal model (BN) on CCTV-style causal data
    logger.info("[1] Training causal model (Bayesian Network, Sedlak 2024 style)...")
    training_data = generate_cctv_causal_training_data(150)
    bn_learner = BayesianNetworkLearner()
    eosc_model = bn_learner.learn_structure(training_data)
    eosc_model = bn_learner.update_parameters(eosc_model, training_data)
    logger.success("Causal model ready (bandwidth_low / node_overload → symptoms → slo_violated)")
    logger.info("")

    # 2. Multi-agent system with SLO-aligned thresholds
    logger.info("[2] Building multi-agent system (parallel micro-agents)...")
    detector = DetectorAgent(
        "detector-001",
        thresholds={
            'fps_bad': 0.5,
            'delay_bad': 0.5,
            'throughput_bad': 0.5,
            'slo_violated': 0.5,
        }
    )
    rca_agents = [
        RootCauseAgent("rca-network", eosc_model),
        RootCauseAgent("rca-performance", eosc_model),
    ]
    planner = SolutionPlannerAgent("planner-001", eosc_model)
    executor = ExecutorAgent("executor-001", eosc_model, vfe_threshold=2.0)
    # Use lower threshold only for demo so mitigation runs (offload_to_fog); default 0.85 for safety
    certainty_threshold = 0.55 if os.environ.get("DEMO_MITIGATION") == "1" else 0.85
    if certainty_threshold < 0.85:
        logger.info("DEMO_MITIGATION=1: using lower certainty threshold to show mitigation")
    certainty_agent = CertaintyCheckAgent(
        agent_id="certainty-001",
        eosc_model=eosc_model,
        certainty_threshold=certainty_threshold,
        vfe_threshold=2.0,
    )
    coordinator = MultiAgentCoordinator(device_id="raspberry-pi-001", certainty_threshold=certainty_threshold)
    coordinator.register_agents(detector, rca_agents, planner, executor, certainty_check_agent=certainty_agent)
    logger.success("Agents ready (Detector, 2× RCA, Planner, Executor)")
    logger.info("")

    # 3. Run pipeline on congestion observation
    logger.info("[3] Running pipeline: detect → root cause (high certainty) → mitigate...")
    result = coordinator.process_observation(congestion)

    # 4. Results
    logger.info("")
    logger.info("=" * 72)
    logger.info("REAL-WORLD SCENARIO RESULT")
    logger.info("=" * 72)
    logger.info("Root cause identified: {}", result.get('root_cause', 'N/A'))
    logger.info("Certainty: {}", result.get('certainty', 0))
    vfe = result.get('vfe')
    logger.info("VFE: {} (act only if < threshold)", vfe if vfe is not None else "N/A")
    logger.info("Action taken: {}", result.get('action_taken', 'None'))
    logger.info("Abstained: {} {}", result.get('abstained', False), " (safe: no action under uncertainty)" if result.get('abstained') else "")
    if result.get('reason'):
        logger.info("Reason: {}", result.get('reason'))
    if result.get('parallel_findings'):
        issues = [f for f in result['parallel_findings'] if f.issue_detected]
        logger.info("Parallel agents: {} issue(s) detected (bandwidth/camera/node/delay)", len(issues))
        for f in issues:
            logger.info("  • {}: {} → {}", f.agent_id, f.issue_type, f.recommendation)
    logger.info("=" * 72)
    logger.success("Real-world scenario test complete.")
    return result


if __name__ == "__main__":
    main()
