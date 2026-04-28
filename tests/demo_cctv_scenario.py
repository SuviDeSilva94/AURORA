"""
CCTV Scenario Demo - Raspberry Pi processing live CCTV with network congestion.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aif.multi_agent_system import MultiAgentCoordinator, DetectorAgent, RootCauseAgent, SolutionPlannerAgent, ExecutorAgent
from src.aif.bayesian_network import BayesianNetworkLearner
from src.aif.specialized_agents import ParallelCheckCoordinator
from loguru import logger
import random


def generate_cctv_training_data(num_samples=100):
    """Generate CCTV camera-specific training data"""
    logger.info(f"Generating {num_samples} CCTV training samples...")
    
    data = []
    for _ in range(num_samples):
        # CCTV metrics
        fps = random.choice([30, 28, 25, 20, 15, 35, 40])  # FPS (some excessive)
        delay = random.choice([20, 33, 50, 80, 120])  # Network delay (ms)
        throughput = random.choice([2.5, 1.8, 1.4, 0.9, 0.5])  # Throughput (MB/s)
        cpu = random.uniform(30, 95)  # CPU %
        memory = random.uniform(40, 90)  # Memory %
        
        # Root causes (causal relationships)
        camera_excessive = fps > 32  # Camera sending too many frames
        bandwidth_low = throughput < 1.6  # Network congestion
        node_overloaded = cpu > 85 or memory > 85  # Node overload
        
        # SLO violation (fps < 30 OR delay > 33ms OR throughput < 1.6 MB/s)
        slo_violated = (fps < 28 or delay > 35 or throughput < 1.6)
        
        # Discretize for Bayesian Network
        data.append({
            'fps': 'low' if fps < 28 else ('high' if fps > 32 else 'normal'),
            'delay': 'high' if delay > 35 else 'normal',
            'throughput': 'low' if throughput < 1.6 else 'normal',
            'cpu': 'high' if cpu > 70 else 'normal',
            'memory': 'high' if memory > 75 else 'normal',
            'camera_excessive': camera_excessive,
            'bandwidth_low': bandwidth_low,
            'node_overloaded': node_overloaded,
            'slo_violated': slo_violated
        })
    
    logger.success(f"Generated {len(data)} samples")
    return data


def run_cctv_demo():
    """
    Run CCTV scenario demo
    Exact scenario from  :
    - Raspberry Pi processing live CCTV camera footage
    - Sudden network congestion → FPS drop, delay spike to 80ms, throughput drop
    """
    logger.info("=" * 70)
    logger.info("CCTV SECURITY CAMERA SCENARIO -   REQUIREMENTS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Setup:")
    logger.info("  - Raspberry Pi 4 (edge node)")
    logger.info("  - 4 CCTV cameras (3× 1080p, 1× 4K)")
    logger.info("  - Target SLOs: 30 FPS, <33ms delay, >1.6 MB/s throughput")
    logger.info("")
    logger.info("Scenario: Network congestion detected!")
    logger.info("=" * 70)
    
    # Step 1: Train causal model
    logger.info("\n[TRAINING] Learning causal model from historical data...")
    training_data = generate_cctv_training_data(100)
    
    bn_learner = BayesianNetworkLearner()
    eosc_model = bn_learner.learn_structure(training_data)
    eosc_model = bn_learner.update_parameters(eosc_model, training_data)
    logger.success("Causal model learned (Bayesian Network)")
    
    # Step 2: Create multi-agent system
    logger.info("\n[SETUP] Creating distributed micro-agents...")
    
    # Create individual agents first
    detector = DetectorAgent(
        "detector-001",
        thresholds={'fps': 28, 'delay': 35, 'throughput': 1.5, 'cpu': 80, 'memory': 85}
    )
    
    rca_agents = [
        RootCauseAgent("rca-network-001", eosc_model),
        RootCauseAgent("rca-performance-001", eosc_model)
    ]
    
    planner = SolutionPlannerAgent("planner-001", eosc_model)
    executor = ExecutorAgent("executor-001", eosc_model, vfe_threshold=2.0)
    
    # Create coordinator and register agents
    coordinator = MultiAgentCoordinator(device_id="raspberry-pi-001")
    coordinator.detector = detector
    coordinator.rca_agents = rca_agents
    coordinator.planner = planner
    coordinator.executor = executor
    coordinator.certainty_threshold = 0.70  # Lower threshold for demo
    
    logger.success("Multi-agent system ready")
    
    # Step 3: Simulate CCTV congestion scenario
    logger.info("\n[SIMULATION] Injecting network congestion fault...")
    
    observation = {
        'fps': 18,              # FPS dropped from 30 to 18
        'delay': 80,            # Delay spiked to 80ms (target: <33ms)
        'throughput': 0.9,      # Throughput dropped to 0.9 MB/s (target: >1.6)
        'cpu': 75,              # CPU at 75%
        'memory': 65,           # Memory at 65%
        'camera_id': 'camera_3',
        'node_id': 'raspberry-pi-001',
        'slo_violated': True
    }
    
    logger.warning("Congestion detected:")
    logger.warning(f"  FPS: {observation['fps']} (target: 30)")
    logger.warning(f"  Delay: {observation['delay']}ms (target: <33ms)")
    logger.warning(f"  Throughput: {observation['throughput']} MB/s (target: >1.6)")
    
    # Step 4: Process with multi-agent system
    logger.info("\n[PROCESSING] Multi-agent self-healing pipeline...")
    result = coordinator.process_observation(observation)
    
    # Step 5: Show results
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Root Cause Identified: {result.get('root_cause', 'None')}")
    logger.info(f"Certainty: {result.get('certainty', 0)*100:.1f}%")
    vfe_value = result.get('vfe')
    logger.info(f"VFE: {vfe_value:.2f}" if vfe_value is not None else "VFE: N/A")
    logger.info(f"Action Taken: {result.get('action_taken', 'None')}")
    logger.info(f"Abstained: {result.get('abstained', False)}")
    
    if result.get('abstained'):
        logger.warning(f"Reason: {result.get('reason', 'unknown')}")
    
    if result.get('parallel_findings'):
        logger.info(f"\nParallel Agent Findings: {len(result.get('parallel_findings', []))} agents checked")
        for finding in result.get('parallel_findings', []):
            if finding.issue_detected:
                logger.info(f"  - {finding.agent_id}: {finding.issue_type} (confidence={finding.confidence*100:.1f}%)")
                logger.info(f"    Recommendation: {finding.recommendation}")
    
    logger.info("=" * 70)
    logger.info("\n✅ Demo Complete! The system:")
    logger.info("  1. Trained on CCTV-specific data (100 samples)")
    logger.info("  2. Ran parallel specialized agents (4 agents)")
    logger.info("  3. Detected anomalies (delay spike)")
    logger.info("  4. Attempted root cause analysis")
    logger.info("  5. Demonstrated two-step safety mechanism")
    logger.info("\nNote: Root cause identification requires sufficient training data")
    logger.info("with clear causal relationships for the Bayesian Network to learn.")
    logger.info("In production, use more training data and domain-specific features.")
    
    return result


if __name__ == "__main__":
    result = run_cctv_demo()

