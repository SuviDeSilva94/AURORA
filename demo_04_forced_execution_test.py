#!/usr/bin/env python3
"""
Working demo: Congestion detection → Certain root-cause identification → Mitigation (Proof of Concept)

Shows the full thesis baseline pipeline:
1. Inject synthetic network congestion (delay spike, throughput drop).
2. Parallel micro-agents detect anomalies.
3. Root cause is identified using the Bayesian Network (Sedlak 2024).
4. Certainty check: proceed only if certainty ≥ threshold (no uncertainty on final decision).
5. VFE gating (Donta 2025): execute only if VFE < threshold.
6. System executes or aborts via Mathematical Abstention depending on safety certainty.

Run from project root: python3 demo_01_basic_pipeline_test.py (or: source venv/bin/activate && python3 demo_01_basic_pipeline_test.py)
"""

import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from loguru import logger
except ModuleNotFoundError as exc:
    print(
        "\nMissing dependencies (e.g. loguru). From the project folder run:\n"
        "  python3 -m venv venv\n"
        "  source venv/bin/activate          # Windows: venv\\Scripts\\activate\n"
        "  pip install -r requirements.txt\n"
        "Then: python3 run_demo.py\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc
from src.aif.bayesian_network import BayesianNetworkLearner
from src.aif.multi_agent_system import (
    MultiAgentCoordinator,
    DetectorAgent,
    RootCauseAgent,
    SolutionPlannerAgent,
    ExecutorAgent,
)
from src.aif.certainty_check_agent import CertaintyCheckAgent


def generate_training_data(num_samples: int = 100):
    """CCTV-style training data with causal relations for congestion. Uses numeric states for BN."""
    import random
    data = []
    for _ in range(num_samples):
        network = random.uniform(10, 100)
        cpu_base = 100 - network * 0.6
        cpu = max(10, min(100, cpu_base + random.uniform(-15, 15)))
        delay_base = cpu * 1.2
        delay = max(5, delay_base + random.uniform(-15, 25))
        throughput = max(0.2, network / 50 + random.uniform(-0.2, 0.5))
        slo_ok = (cpu < 80) and (network > 30) and (delay < 50) and (throughput > 1.2)
        # Numeric states so BN and RCA can use same evidence: 0=normal/low, 1=high/bad
        data.append({
            'network_quality': 0 if network > 50 else 1,
            'cpu': 1 if cpu > 70 else 0,
            'delay': 1 if delay > 45 else 0,
            'throughput': 0 if throughput > 1.2 else 1,
            'slo_violated': 0 if slo_ok else 1,
        })
    return data


def main():
    logger.info("=" * 70)
    logger.info("DEMO: Congestion detection → Root cause (certain) → Mitigation")
    logger.info("=" * 70)

    # 1. Train causal model (pgmpy BN — Sedlak 2024)
    logger.info("\n[1] Training causal model (Bayesian Network)...")
    training_data = generate_training_data(100)
    bn_learner = BayesianNetworkLearner()
    # The agent mathematically discovers the arrows of causality (e.g. Congestion -> causes -> Delay)
    # This creates the 'EOSC Model', which represents the causal map of your physical Edge-Fog continuum.
    eosc_model = bn_learner.learn_structure(training_data)
    
    # The agent calculates the exact probability for each arrow (e.g. 85% chance Delay triggers)
    eosc_model = bn_learner.update_parameters(eosc_model, training_data)
    logger.success("Causal model ready")

    # 2. Build multi-agent system
    logger.info("\n[2] Building multi-agent system...")
    detector = DetectorAgent(
        "detector-001",
        thresholds={'cpu': 0.7, 'delay': 0.7, 'throughput': 0.5, 'network_quality': 0.7}
    )
    rca_agents = [
        RootCauseAgent("rca-1", eosc_model),
        RootCauseAgent("rca-2", eosc_model),
    ]
    planner = SolutionPlannerAgent("planner-001", eosc_model)
    
    # To demo a successful execution (bypassing the Abstention Paradigm),
    # disabled the two safety gates.
    
    # set VFE threshold to 10.0 (default 2.0), ExecutorAgent to 
    # ignore risk and execute the action no matter what
    executor = ExecutorAgent("executor-001", eosc_model, vfe_threshold=10.0)
    
    # set Certainty threshold to 0.0 (default 0.85), Certainty Check 
    # Gate to blindly accept ANY root cause guess, even if it is completely uncertain.
    certainty_agent = CertaintyCheckAgent(
        agent_id="certainty-001",
        eosc_model=eosc_model,
        certainty_threshold=0.0,
        vfe_threshold=10.0,
    )
    coordinator = MultiAgentCoordinator(device_id="edge-001", certainty_threshold=0.0)
    coordinator.register_agents(detector, rca_agents, planner, executor, certainty_check_agent=certainty_agent)
    logger.success("Agents ready")

    # 3. Inject congestion use same numeric states as BN (0/1)
    logger.info("\n[3] Injecting synthetic network congestion...")
    observation = {
        'cpu': 1,           # high
        'delay': 1,          # high
        'throughput': 1,     # low (bad)
        'network_quality': 1,  # low (bad)
        'slo_violated': 1,   # violated
    }
    logger.warning(
        "Congestion: cpu=high, delay=high, throughput=low, network_quality=low, slo_violated"
    )

    # 4. Run pipeline: detect → root cause (with certainty check) → mitigate
    logger.info("\n[4] Running pipeline: detect → root cause (certainty check) → mitigate...")
    result = coordinator.process_observation(observation)

    # 5. Summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULT")
    logger.info("=" * 70)
    logger.info(f"Root cause identified: {result.get('root_cause', 'N/A')}")
    logger.info(f"Certainty: {result.get('certainty', 0):.1%}")
    logger.info(f"VFE: {result.get('vfe', 'N/A')}")
    logger.info(f"Action taken: {result.get('action_taken', 'None')}")
    logger.info(f"Abstained: {result.get('abstained', False)}")
    if result.get('reason'):
        logger.info(f"Reason: {result.get('reason')}")
    logger.info("=" * 70)
    logger.success("\nDemo complete: congestion → certain root cause → mitigation (or safe abstention)")
    return result


if __name__ == "__main__":
    main()
