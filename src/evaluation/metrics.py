"""
Metrics Collection Framework for Experimental Evaluation

This module tracks key metrics required to answer RQ3:
- Repair Accuracy: Did the agent fix the correct root cause?
- Abstention Rate: How often did the agent refuse to act under uncertainty?
- Destructive Actions: How many harmful actions were taken?
- Mean Time to Repair (MTTR): How long did recovery take?
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from pathlib import Path
from loguru import logger


class ActionOutcome(Enum):
    """Possible outcomes of an agent's action"""
    CORRECT = "correct"           # Fixed the right root cause
    INCORRECT = "incorrect"       # Fixed wrong thing
    DESTRUCTIVE = "destructive"   # Made things worse
    ABSTAINED = "abstained"       # Refused to act (uncertainty too high)
    NO_ACTION = "no_action"       # No action needed


@dataclass
class TrialMetrics:
    """Metrics collected from a single experimental trial"""
    
    # Trial identification
    trial_id: int
    agent_type: str
    fault_type: str
    timestamp: float
    
    # Core metrics
    repair_accuracy: float  # 0.0 to 1.0 (1.0 = perfect repair)
    abstention: bool        # True if agent abstained from action
    destructive_action: bool  # True if action made things worse
    mttr_seconds: float     # Time to repair (detection → recovery)
    
    # Detailed outcome
    action_outcome: str  # ActionOutcome enum value
    root_cause_identified: Optional[str] = None
    action_taken: Optional[str] = None
    
    # Additional context
    initial_state: Dict[str, Any] = field(default_factory=dict)
    final_state: Dict[str, Any] = field(default_factory=dict)
    vfe_value: Optional[float] = None
    certainty_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple trials"""
    
    agent_type: str
    num_trials: int
    
    # Mean values
    mean_repair_accuracy: float
    mean_mttr_seconds: float
    resolution_rate: float  # (correct + no_action) / num_trials
    abstention_rate: float  # Percentage of trials where agent abstained
    destructive_action_rate: float  # Percentage of trials with destructive actions
    
    # Standard deviations
    std_repair_accuracy: float
    std_mttr_seconds: float
    
    # Counts
    num_correct: int
    num_incorrect: int
    num_destructive: int
    num_abstained: int
    num_no_action: int
    
    # By fault type
    metrics_by_fault: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class MetricsCollector:
    """
    Collects and tracks metrics during experimental trials
    
    Usage:
        collector = MetricsCollector()
        
        # Start trial
        collector.start_trial(trial_id=1, agent_type="AURORA", fault_type="network_drop")
        
        # ... agent operates ...
        
        # End trial
        collector.end_trial(
            action_outcome=ActionOutcome.CORRECT,
            root_cause="network_quality",
            action_taken="offload_to_fog"
        )
        
        # Save results
        collector.save_results("results/experiment_1.json")
    """
    
    def __init__(self):
        self.current_trial: Optional[Dict[str, Any]] = None
        self.trial_metrics: List[TrialMetrics] = []
        
    def start_trial(
        self,
        trial_id: int,
        agent_type: str,
        fault_type: str,
        initial_state: Optional[Dict[str, Any]] = None
    ):
        """Start tracking a new trial"""
        self.current_trial = {
            'trial_id': trial_id,
            'agent_type': agent_type,
            'fault_type': fault_type,
            'start_time': time.time(),
            'initial_state': initial_state or {}
        }
        logger.debug(f"Started trial {trial_id}: {agent_type} on {fault_type}")
    
    def end_trial(
        self,
        action_outcome: ActionOutcome,
        root_cause_identified: Optional[str] = None,
        action_taken: Optional[str] = None,
        final_state: Optional[Dict[str, Any]] = None,
        vfe_value: Optional[float] = None,
        certainty_score: Optional[float] = None,
        ground_truth_root_cause: Optional[str] = None
    ):
        """End the current trial and compute metrics"""
        if self.current_trial is None:
            raise ValueError("No trial in progress. Call start_trial() first.")
        
        end_time = time.time()
        mttr = end_time - self.current_trial['start_time']
        
        # Calculate repair accuracy
        repair_accuracy = self._calculate_repair_accuracy(
            action_outcome=action_outcome,
            root_cause_identified=root_cause_identified,
            ground_truth_root_cause=ground_truth_root_cause
        )
        
        # Determine if action was destructive
        is_destructive = (action_outcome == ActionOutcome.DESTRUCTIVE)
        
        # Determine if agent abstained
        did_abstain = (action_outcome == ActionOutcome.ABSTAINED)
        
        # Create trial metrics
        metrics = TrialMetrics(
            trial_id=self.current_trial['trial_id'],
            agent_type=self.current_trial['agent_type'],
            fault_type=self.current_trial['fault_type'],
            timestamp=self.current_trial['start_time'],
            repair_accuracy=repair_accuracy,
            abstention=did_abstain,
            destructive_action=is_destructive,
            mttr_seconds=mttr,
            action_outcome=action_outcome.value,
            root_cause_identified=root_cause_identified,
            action_taken=action_taken,
            initial_state=self.current_trial['initial_state'],
            final_state=final_state or {},
            vfe_value=vfe_value,
            certainty_score=certainty_score
        )
        
        self.trial_metrics.append(metrics)
        logger.info(
            f"Trial {metrics.trial_id} completed: "
            f"Accuracy={repair_accuracy:.2f}, MTTR={mttr:.1f}s, "
            f"Outcome={action_outcome.value}"
        )
        
        self.current_trial = None
        return metrics
    
    def _calculate_repair_accuracy(
        self,
        action_outcome: ActionOutcome,
        root_cause_identified: Optional[str],
        ground_truth_root_cause: Optional[str]
    ) -> float:
        """
        Calculate repair accuracy (0.0 to 1.0)
        
        Logic:
        - CORRECT outcome = 1.0
        - INCORRECT outcome = 0.5 (tried but wrong)
        - DESTRUCTIVE outcome = 0.0 (made worse)
        - ABSTAINED outcome = 0.0 (didn't act when should have)
        - NO_ACTION outcome = 1.0 (correctly did nothing)
        
        If ground truth is provided, also check if identified root cause matches
        """
        if action_outcome == ActionOutcome.CORRECT:
            accuracy = 1.0
        elif action_outcome == ActionOutcome.NO_ACTION:
            accuracy = 1.0
        elif action_outcome == ActionOutcome.INCORRECT:
            accuracy = 0.3
        elif action_outcome == ActionOutcome.ABSTAINED:
            # Abstaining when uncertain is safe, but doesn't fix the problem
            accuracy = 0.5
        elif action_outcome == ActionOutcome.DESTRUCTIVE:
            accuracy = 0.0
        else:
            accuracy = 0.0
        
        # Bonus: If we can verify root cause identification
        if ground_truth_root_cause and root_cause_identified:
            if root_cause_identified == ground_truth_root_cause:
                accuracy = min(1.0, accuracy + 0.1)  # Bonus for correct identification
            else:
                accuracy *= 0.5  # Penalty for wrong identification
        
        return accuracy
    
    def get_aggregated_metrics(self, agent_type: Optional[str] = None) -> AggregatedMetrics:
        """
        Compute aggregated metrics across trials
        
        Args:
            agent_type: If provided, only aggregate for this agent type
        """
        # Filter trials
        trials = self.trial_metrics
        if agent_type:
            trials = [t for t in trials if t.agent_type == agent_type]
        
        if not trials:
            raise ValueError(f"No trials found for agent_type={agent_type}")
        
        # Calculate aggregates
        num_trials = len(trials)
        accuracies = [t.repair_accuracy for t in trials]
        mttrs = [t.mttr_seconds for t in trials]
        
        num_abstained = sum(1 for t in trials if t.abstention)
        num_destructive = sum(1 for t in trials if t.destructive_action)
        
        num_correct = sum(1 for t in trials if t.action_outcome == ActionOutcome.CORRECT.value)
        num_incorrect = sum(1 for t in trials if t.action_outcome == ActionOutcome.INCORRECT.value)
        num_no_action = sum(1 for t in trials if t.action_outcome == ActionOutcome.NO_ACTION.value)
        
        import statistics
        mean_accuracy = statistics.mean(accuracies)
        std_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        mean_mttr = statistics.mean(mttrs)
        std_mttr = statistics.stdev(mttrs) if len(mttrs) > 1 else 0.0
        
        abstention_rate = (num_abstained / num_trials) * 100
        destructive_rate = (num_destructive / num_trials) * 100
        resolution_rate = (num_correct + num_no_action) / num_trials
        
        # Aggregate by fault type
        metrics_by_fault = {}
        fault_types = set(t.fault_type for t in trials)
        for fault in fault_types:
            fault_trials = [t for t in trials if t.fault_type == fault]
            fault_accuracies = [t.repair_accuracy for t in fault_trials]
            fault_mttrs = [t.mttr_seconds for t in fault_trials]
            metrics_by_fault[fault] = {
                'num_trials': len(fault_trials),
                'mean_accuracy': statistics.mean(fault_accuracies),
                'mean_mttr': statistics.mean(fault_mttrs)
            }
        
        agent_name = agent_type or trials[0].agent_type
        
        return AggregatedMetrics(
            agent_type=agent_name,
            num_trials=num_trials,
            mean_repair_accuracy=mean_accuracy,
            mean_mttr_seconds=mean_mttr,
            resolution_rate=resolution_rate,
            abstention_rate=abstention_rate,
            destructive_action_rate=destructive_rate,
            std_repair_accuracy=std_accuracy,
            std_mttr_seconds=std_mttr,
            num_correct=num_correct,
            num_incorrect=num_incorrect,
            num_destructive=num_destructive,
            num_abstained=num_abstained,
            num_no_action=num_no_action,
            metrics_by_fault=metrics_by_fault
        )
    
    def save_results(self, filepath: str):
        """Save all trial metrics to JSON file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'trials': [t.to_dict() for t in self.trial_metrics],
            'num_trials': len(self.trial_metrics),
            'timestamp': time.time()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.trial_metrics)} trial results to {filepath}")
    
    @staticmethod
    def load_results(filepath: str) -> List[TrialMetrics]:
        """Load trial metrics from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        trials = []
        for trial_dict in data['trials']:
            trials.append(TrialMetrics(**trial_dict))
        
        logger.info(f"Loaded {len(trials)} trial results from {filepath}")
        return trials
    
    def reset(self):
        """Clear all collected metrics"""
        self.trial_metrics = []
        self.current_trial = None
        logger.info("Metrics collector reset")


# Convenience function for quick metric summary
def print_metrics_summary(aggregated: AggregatedMetrics):
    """Print a formatted summary of aggregated metrics"""
    print(f"\n{'='*70}")
    print(f"METRICS SUMMARY: {aggregated.agent_type}")
    print(f"{'='*70}")
    print(f"Total Trials:          {aggregated.num_trials}")
    print(f"")
    print(f"Repair Accuracy:       {aggregated.mean_repair_accuracy:.1%} (±{aggregated.std_repair_accuracy:.1%})")
    print(f"Mean Time to Repair:   {aggregated.mean_mttr_seconds:.2f}s (±{aggregated.std_mttr_seconds:.2f}s)")
    print(f"Resolution Rate:       {aggregated.resolution_rate:.1%}")
    print(f"Abstention Rate:       {aggregated.abstention_rate:.1f}%")
    print(f"Destructive Actions:   {aggregated.destructive_action_rate:.1f}%")
    print(f"")
    print(f"Outcome Breakdown:")
    print(f"  ✓ Correct:           {aggregated.num_correct}")
    print(f"  ✗ Incorrect:         {aggregated.num_incorrect}")
    print(f"  ⚠ Destructive:       {aggregated.num_destructive}")
    print(f"  ⏸ Abstained:         {aggregated.num_abstained}")
    print(f"  - No Action:         {aggregated.num_no_action}")
    
    if aggregated.metrics_by_fault:
        print(f"")
        print(f"By Fault Type:")
        for fault, metrics in aggregated.metrics_by_fault.items():
            print(f"  {fault:20s} Acc={metrics['mean_accuracy']:.1%}, MTTR={metrics['mean_mttr']:.1f}s")
    
    print(f"{'='*70}\n")


