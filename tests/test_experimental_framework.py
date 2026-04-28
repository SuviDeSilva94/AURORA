"""
Experimental framework tests: metrics, baselines, fault injection, runner, analysis.
Run: python tests/test_experimental_framework.py
"""

import sys
import json
import random
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    MetricsCollector,
    ActionOutcome,
    print_metrics_summary
)
from src.baselines.rule_based_agent import RuleBasedAgentWrapper
from src.baselines.aif_no_gate_agent import AIFNoGateAgentWrapper
from src.aif.bayesian_network import fit_eosc_cctv_benchmark_dag
from src.aif.root_cause_analyzer import RootCauseAnalyzer


def test_metrics_collection():
    """Test 1: Metrics Collection Framework"""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Metrics Collection")
    logger.info("="*70)
    
    collector = MetricsCollector()
    
    # Simulate 3 trials
    for trial_id in range(1, 4):
        # Start trial
        collector.start_trial(
            trial_id=trial_id,
            agent_type="test_agent",
            fault_type="test_fault",
            initial_state={'cpu': 50.0}
        )
        
        # Simulate agent action
        import time
        time.sleep(0.1)
        
        # End trial
        outcome = ActionOutcome.CORRECT if trial_id % 2 == 0 else ActionOutcome.INCORRECT
        collector.end_trial(
            action_outcome=outcome,
            root_cause_identified="test_cause",
            action_taken="test_action",
            vfe_value=random.uniform(0.5, 2.5)
        )
    
    # Get aggregated metrics
    aggregated = collector.get_aggregated_metrics()
    print_metrics_summary(aggregated)
    
    # Validate
    assert aggregated.num_trials == 3
    assert aggregated.agent_type == "test_agent"
    assert 0 <= aggregated.mean_repair_accuracy <= 1.0
    assert aggregated.mean_mttr_seconds > 0
    
    logger.success("✓ Metrics collection test passed")


def test_rule_based_agent():
    """Test 2: Rule-Based Baseline Agent"""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Rule-Based Agent")
    logger.info("="*70)
    
    agent = RuleBasedAgentWrapper("test-rule")
    
    # Test normal observation (should not act)
    obs_normal = {
        'cpu': 45.0,
        'memory': 60.0,
        'network': 80.0,
        'delay': 30.0,
        'slo_violated': False
    }
    
    result = agent.process_observation(obs_normal)
    logger.info(f"Normal observation result: {result}")
    assert result['action_taken'] is None
    assert result['abstained'] is False
    
    # Test high CPU observation (should act)
    obs_high_cpu = {
        'cpu': 92.0,
        'memory': 60.0,
        'network': 80.0,
        'delay': 30.0,
        'slo_violated': True
    }
    
    result = agent.process_observation(obs_high_cpu)
    logger.info(f"High CPU observation result: {result}")
    assert result['action_taken'] is not None
    assert result['action_taken'] == 'restart'
    assert result['abstained'] is False
    
    # Test low network observation (should act)
    obs_low_network = {
        'cpu': 45.0,
        'memory': 60.0,
        'network': 15.0,
        'delay': 30.0,
        'slo_violated': True
    }
    
    result = agent.process_observation(obs_low_network)
    logger.info(f"Low network observation result: {result}")
    assert result['action_taken'] is not None
    assert result['action_taken'] == 'offload_to_fog'
    
    logger.success("✓ Rule-based agent test passed")


def test_aif_no_gate_agent():
    """Test 3: AIF-No-Gate Baseline Agent"""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: AIF-No-Gate Agent")
    logger.info("="*70)
    
    agent = AIFNoGateAgentWrapper("test-aif")
    
    # Generate training data
    training_data = []
    for i in range(50):
        network = random.uniform(10, 100)
        cpu = 100 - network * 0.7 + random.uniform(-15, 15)
        cpu = max(10, min(100, cpu))
        slo_violated = (cpu > 80) or (network < 25)
        
        training_data.append({
            'network_quality': 'low' if network < 40 else 'high',
            'cpu': 'high' if cpu > 70 else 'normal',
            'memory': 'normal',
            'slo_violated': slo_violated
        })
    
    # Train agent
    logger.info("Training AIF-no-gate agent...")
    agent.train(training_data)
    logger.info("Training complete")
    
    # Test normal observation
    obs_normal = {
        'cpu': 45.0,
        'memory': 60.0,
        'network': 80.0,
        'network_quality': 'high',
        'delay': 30.0,
        'slo_violated': False
    }
    
    result = agent.process_observation(obs_normal)
    logger.info(f"Normal observation result: {result}")
    assert result['action_taken'] is None
    
    # Test SLO violation
    obs_violation = {
        'cpu': 85.0,
        'memory': 60.0,
        'network': 20.0,
        'network_quality': 'low',
        'delay': 120.0,
        'slo_violated': True
    }
    
    result = agent.process_observation(obs_violation)
    logger.info(f"SLO violation result: {result}")
    # Note: AIF agent might act or not depending on what it learned
    # But it should never abstain (no gating)
    assert result['abstained'] is False
    assert result['vfe'] is not None  # Should compute VFE
    
    logger.success("✓ AIF-no-gate agent test passed")


def test_fault_injection():
    """Test 4: Fault Injection"""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Fault Injection")
    logger.info("="*70)
    
    from experiments.run_comparison import FaultInjector
    
    # Test normal observation
    obs_normal = FaultInjector.generate_normal_observation()
    logger.info(f"Normal observation: {obs_normal}")
    assert obs_normal['slo_violated'] is False
    assert 30 <= obs_normal['cpu'] <= 60
    assert 60 <= obs_normal['network'] <= 100
    
    # Test network drop
    obs_network = FaultInjector.inject_network_drop(severity=0.8)
    logger.info(f"Network drop: {obs_network}")
    assert obs_network['slo_violated'] is True
    assert obs_network['network'] < 30
    assert obs_network['network_quality'] == 'low'
    
    # Test CPU spike
    obs_cpu = FaultInjector.inject_cpu_spike(severity=0.8)
    logger.info(f"CPU spike: {obs_cpu}")
    assert obs_cpu['slo_violated'] is True
    assert obs_cpu['cpu'] > 85
    
    # Test memory leak
    obs_memory = FaultInjector.inject_memory_leak(severity=0.8)
    logger.info(f"Memory leak: {obs_memory}")
    assert obs_memory['slo_violated'] is True
    assert obs_memory['memory'] > 85
    
    logger.success("✓ Fault injection test passed")


def test_memory_leak_root_cause_top2_contains_memory():
    """Memory-leak-like observation should include memory in top-2 causes."""
    from experiments.run_comparison import ExperimentRunner

    runner = ExperimentRunner(num_trials=1, output_dir="experiments/results_test")
    training = runner.generate_training_data(num_samples=1200)
    model = fit_eosc_cctv_benchmark_dag(training)
    rca = RootCauseAnalyzer(model)

    obs = {
        "network_quality": "medium",
        "cpu": "high",
        "memory": "high",
        "delay": "high",
        "slo_violated": True,
    }
    causes = rca.find_root_causes(
        symptom="slo_violated",
        symptom_value=True,
        observation=obs,
        top_k=2,
    )
    top2 = [c.variable for c in causes[:2]]
    assert "memory" in top2, f"Expected memory in top-2, got {top2}"


def test_mini_experiment():
    """Test 5: Mini Experiment (3 trials)"""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Mini Experiment Runner")
    logger.info("="*70)
    
    from experiments.run_comparison import ExperimentRunner
    
    # Run mini experiment with just 2 trials per fault
    logger.info("Running mini experiment (2 trials × 3 faults = 6 trials per agent)...")
    
    runner = ExperimentRunner(
        num_trials=2,
        output_dir="experiments/results_test"
    )
    
    # This will run the full comparison but with fewer trials
    results = runner.run_experiment()
    
    # Validate results
    assert 'rule_based' in results
    assert 'aif_no_gate' in results
    assert 'aurora' in results
    
    for agent_name, aggregated in results.items():
        logger.info(f"\n{agent_name}:")
        logger.info(f"  Trials: {aggregated.num_trials}")
        logger.info(f"  Accuracy: {aggregated.mean_repair_accuracy:.1%}")
        logger.info(f"  MTTR: {aggregated.mean_mttr_seconds:.2f}s")
        logger.info(f"  Abstention: {aggregated.abstention_rate:.1f}%")
        
        assert aggregated.num_trials == 6  # 2 trials × 3 fault types
        assert 0 <= aggregated.mean_repair_accuracy <= 1.0
        assert aggregated.mean_mttr_seconds > 0
    
    # Validate files were created
    results_dir = Path("experiments/results_test")
    assert (results_dir / "rule_based_results.json").exists()
    assert (results_dir / "aif_no_gate_results.json").exists()
    assert (results_dir / "aurora_results.json").exists()
    assert (results_dir / "comparison_summary.json").exists()

    with open(results_dir / "aurora_trial_diagnostics.json") as f:
        aurora_diag = json.load(f)
    assert len(aurora_diag) == 6
    allowed = {
        "none",
        "no_slo_violation",
        "no_root_causes",
        "low_certainty",
        "high_vfe",
        "ambiguous",
        "symphony_disagree",
    }
    for row in aurora_diag:
        assert row.get("abstain_reason") in allowed, row
    
    logger.success("✓ Mini experiment test passed")


def test_results_analysis():
    """Test 6: Results Analysis"""
    logger.info("\n" + "="*70)
    logger.info("TEST 6: Results Analysis")
    logger.info("="*70)
    
    from experiments.analyze_results import ResultsAnalyzer
    
    # Analyze mini experiment results
    analyzer = ResultsAnalyzer(results_dir="experiments/results_test")
    
    # Create comparison table
    df = analyzer.create_comparison_table()
    logger.info("\nComparison Table:")
    print(df)
    
    assert len(df) == 3  # 3 agents
    assert 'Agent Type' in df.columns
    assert 'Repair Accuracy (mean trial score)' in df.columns
    assert 'Repair correct (Wilson 95%)' in df.columns
    assert 'MTTR (seconds)' in df.columns

    ci = analyzer.compute_all_confidence_intervals()
    assert "aurora" in ci["agents"]
    assert "wilson" in ci["agents"]["aurora"]

    logger.success("✓ Results analysis test passed")


def run_all_tests():
    """Run all tests in sequence"""
    logger.info("\n" + "="*70)
    logger.info("COMPREHENSIVE TEST SUITE - EXPERIMENTAL FRAMEWORK")
    logger.info("="*70)
    
    try:
        test_metrics_collection()
        test_rule_based_agent()
        test_aif_no_gate_agent()
        test_fault_injection()
        test_mini_experiment()
        test_results_analysis()
        
        logger.success("\n" + "="*70)
        logger.success("✅ ALL TESTS PASSED!")
        logger.success("="*70)
        logger.success("\nYour experimental framework is ready to use!")
        logger.success("\nNext steps:")
        logger.success("  1. Run full experiment: python experiments/run_comparison.py")
        logger.success("  2. Analyze results: python experiments/analyze_results.py")
        logger.success("  3. Use results in thesis Chapter 4 (Results)")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)


