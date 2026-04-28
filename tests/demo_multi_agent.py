#!/usr/bin/env python3
"""
Demo: Multi-Agent Root Cause Discovery System.

Pipeline: Detector → RCA agents → Planner → Executor (with VFE gating).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from src.aif.bayesian_network import BayesianNetworkLearner
from src.aif.multi_agent_system import (
    DetectorAgent,
    RootCauseAgent,
    SolutionPlannerAgent,
    ExecutorAgent,
    MultiAgentCoordinator,
)
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {message}", level="INFO")


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_multi_agent_root_cause_discovery():
    """
    Complete multi-agent system for root cause discovery
    """
    
    print_section("MULTI-AGENT ROOT CAUSE DISCOVERY SYSTEM")
    
    print("""
This demonstrates the complete multi-agent pipeline:

   Detector → Multiple RCA Agents → Solution Planner → Executor
                  ↓
          (Active Inference + Bayesian)
    """)
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: TRAINING - Active Inference learns causal models
    # ═══════════════════════════════════════════════════════════════════
    print_section("PHASE 1: Training (Active Inference Learning)")
    
    print("Agents are learning causal models from observations...")
    print("Collecting 60 observations...\n")
    
    training_data = []
    for i in range(60):
        # Simulate normal and faulty conditions
        network_quality = np.random.randint(20, 90)
        cpu = 25 + (100 - network_quality) // 3
        memory = 30 + (100 - network_quality) // 4
        delay = 20 + cpu // 2
        slo_violated = 1 if delay > 50 else 0
        
        training_data.append({
            'network_quality': int(network_quality),
            'cpu': int(cpu),
            'memory': int(memory),
            'delay': int(delay),
            'slo_violated': slo_violated
        })
    
    print("🧠 Active Inference: Learning causal model (Bayesian Network)...")
    learner = BayesianNetworkLearner(algorithm='hc')
    eosc_model = learner.learn_structure(training_data)
    
    edges = eosc_model.get_structure()
    
    print("\n✓ Learned Causal Structure:")
    if edges:
        for parent, child in edges:
            print(f"    {parent} → {child}")
        print("\n  This is the CAUSAL MODEL learned by Active Inference!")
    else:
        print("    (Using baseline model)")
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: CREATE MULTI-AGENT SYSTEM
    # ═══════════════════════════════════════════════════════════════════
    print_section("PHASE 2: Creating Multi-Agent System")
    
    # Create detector agent
    detector = DetectorAgent(
        device_id="camera-001",
        thresholds={
            'network_quality': 30,
            'cpu': 70,
            'memory': 75,
            'delay': 50,
            'slo_violated': 0.5
        }
    )
    
    # Create multiple RCA agents (can analyze different aspects in parallel)
    rca_agent_1 = RootCauseAgent("RCA-network", eosc_model)
    rca_agent_2 = RootCauseAgent("RCA-performance", eosc_model)
    rca_agents = [rca_agent_1, rca_agent_2]
    
    # Create solution planner agent
    planner = SolutionPlannerAgent("planner-001", eosc_model)
    
    # Create executor agent (with VFE gating)
    executor = ExecutorAgent("executor-001", eosc_model, vfe_threshold=2.0)
    
    # Create coordinator
    coordinator = MultiAgentCoordinator("camera-001")
    coordinator.register_agents(detector, rca_agents, planner, executor)
    
    print("\n✓ Multi-agent system initialized:")
    print(f"  • 1 Detector Agent")
    print(f"  • {len(rca_agents)} Root Cause Agents")
    print(f"  • 1 Solution Planner Agent")
    print(f"  • 1 Executor Agent (VFE-gated)")
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: NORMAL OPERATION
    # ═══════════════════════════════════════════════════════════════════
    print_section("PHASE 3: Normal Operation")
    
    normal_obs = {
        'network_quality': 75,
        'cpu': 35,
        'memory': 40,
        'delay': 28,
        'slo_violated': 0
    }
    
    print("Current observation:")
    for k, v in normal_obs.items():
        print(f"  {k}: {v} ✓")
    
    print("\nProcessing...")
    result = coordinator.process_observation(normal_obs)
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: FAULT OCCURS
    # ═══════════════════════════════════════════════════════════════════
    print_section("PHASE 4: Fault Occurs - Multi-Agent Pipeline Activated")
    
    print("🔥 FAULT INJECTED: Network degradation!\n")
    
    faulty_obs = {
        'network_quality': 15,   # ← ROOT CAUSE!
        'cpu': 85,               # ← Symptom
        'memory': 78,            # ← Symptom
        'delay': 92,             # ← Symptom
        'slo_violated': 1        # ← Symptom
    }
    
    print("New observation:")
    for k, v in faulty_obs.items():
        status = "🔴" if (k == 'slo_violated' and v == 1) or (k != 'slo_violated' and v > 70) or (k == 'network_quality' and v < 30) else "🟢"
        print(f"  {k}: {v} {status}")
    
    print("\n" + "─" * 70)
    print("Running Multi-Agent Pipeline...")
    print("─" * 70 + "\n")
    
    # Run the complete pipeline
    result = coordinator.process_observation(faulty_obs)
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 5: SHOW RESULTS
    # ═══════════════════════════════════════════════════════════════════
    print_section("PHASE 5: Results Summary")
    
    print("📊 MULTI-AGENT PIPELINE RESULTS:\n")
    
    print(f"1. DETECTOR AGENT:")
    print(f"   Detected {len(result.get('anomalies', []))} anomalies")
    for anomaly in result.get('anomalies', [])[:3]:
        print(f"     • {anomaly.metric}={anomaly.value} (severity={anomaly.severity:.2f})")
    
    print(f"\n2. ROOT CAUSE AGENTS ({len(rca_agents)} agents):")
    print(f"   Found {len(result.get('root_causes', []))} unique root causes:")
    for rc in result.get('root_causes', [])[:3]:
        print(f"     • {rc.variable}={rc.value}")
        print(f"       Impact: {rc.impact_score:.3f}")
        print(f"       Evidence: {rc.evidence_strength:.2f}")
    
    if result.get('root_causes'):
        print(f"\n   🎯 PRIMARY ROOT CAUSE: {result['root_causes'][0].variable}")
        print(f"      (Not a symptom - this is the ORIGIN!)")
    
    print(f"\n3. SOLUTION PLANNER AGENT:")
    print(f"   Proposed {len(result.get('solutions', []))} solutions:")
    for sol in result.get('solutions', [])[:3]:
        print(f"     • {sol.action} for {sol.root_cause}")
        print(f"       Expected impact: {sol.expected_impact:.2f}")
    
    print(f"\n4. EXECUTOR AGENT (VFE-gated):")
    print(f"   Executed {len(result.get('executed', []))} actions:")
    for ex in result.get('executed', [])[:3]:
        print(f"     • {ex['action']} (status: {ex['status']})")
    
    print(f"\n   Abstentions: {len(result.get('solutions', [])) - len(result.get('executed', []))}")
    print(f"   (Agent abstained when VFE too high - preventing unsafe actions!)")
    
    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print_section("SUMMARY: Multi-Agent System Advantages")
    
    print("""
✅ SPECIALIZATION:
  • Each micro-agent has ONE job (lightweight, focused)
  • Can run on resource-constrained IoT devices
  
✅ PARALLEL PROCESSING:
  • Multiple RCA agents analyze simultaneously
  • If agents agree on root cause → higher confidence!
  
✅ CAUSAL REASONING:
  • Active Inference learns the causal model
  • Root cause analysis uses this causal structure
  • Solution planner uses counterfactual reasoning
  
✅ UNCERTAINTY GATING:
  • Executor only acts when VFE < threshold
  • Prevents cascading failures from uncertain decisions
  
✅ DISTRIBUTED:
  • Each agent runs locally (no central orchestrator needed)
  • Fog layer aggregates findings
  • Scales to large systems

KEY INSIGHT:
  Traditional: One monolithic agent (complex, slow)
  Your System: Multiple specialized micro-agents (simple, fast, distributed)
  
  This IS the core of your thesis! 🎓
""")


def demo_distributed_scenario():
    """
    Show how multiple devices coordinate
    """
    print_section("BONUS: Distributed Multi-Device Scenario")
    
    print("""
Scenario: 3 IoT cameras, each with local multi-agent system

Camera 1: Detects network issue → RCA finds: network_quality=15
Camera 2: Detects network issue → RCA finds: network_quality=18
Camera 3: Detects network issue → RCA finds: network_quality=12

Fog Coordinator:
  • Receives 3 reports, all agree: network is root cause
  • Confidence boosted (multiple agents confirm!)
  • Further analysis: Faulty load balancer (system-wide root cause)
  • Action: Replace load balancer (fixes all cameras!)

This is DISTRIBUTED ROOT CAUSE DISCOVERY across the Computing Continuum! 🌐
""")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  MULTI-AGENT ROOT CAUSE DISCOVERY SYSTEM")
    print("  Each micro-agent has a specialized role")
    print("=" * 70)
    
    demo_multi_agent_root_cause_discovery()
    demo_distributed_scenario()
    
    print("\n" + "=" * 70)
    print("  YOUR THESIS ARCHITECTURE")
    print("=" * 70)
    print("""
Multiple Specialized Micro-Agents:

1. Detector Agent(s)
   └─→ Lightweight anomaly detection on IoT devices

2. Root Cause Agent(s) ← CORE CONTRIBUTION!
   └─→ Use Active Inference + Bayesian to find WHY
   └─→ Multiple agents can analyze in parallel
   └─→ Aggregate findings for higher confidence

3. Solution Planner Agent(s)
   └─→ Propose fixes for each root cause
   └─→ Use counterfactual reasoning

4. Executor Agent(s)
   └─→ Implement solutions
   └─→ VFE-gated (only act when confident) ← NOVEL!

This is NOT one agent - it's a SYSTEM of specialized micro-agents! 🤖

Next: Deploy on real IoT devices, K8s cluster, run experiments! 🚀
""")


