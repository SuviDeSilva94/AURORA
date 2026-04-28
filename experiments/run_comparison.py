"""
Experiment Runner for Comparing AURORA vs Baselines

This script runs controlled experiments to answer RQ3:
"What measurable improvements in repair accuracy, abstention rate under 
uncertainty, and MTTR does AURORA demonstrate compared to baselines?"

Experimental Design:
- 3 agent types: Rule-based, AIF-no-gate, AURORA
- 3 fault types: network_drop, cpu_spike, memory_leak
- 10-20 trials per combination (total: 90-180 trials)
- Metrics: Repair accuracy, MTTR, abstention rate, destructive actions
"""

import random
import time
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json
from loguru import logger

# Import agents
from src.baselines.rule_based_agent import RuleBasedAgentWrapper
from src.baselines.aif_no_gate_agent import AIFNoGateAgentWrapper
from src.aif.multi_agent_certainty import (
    MultiAgentCoordinatorWithCertainty,
)
from src.aif.bayesian_network import (
    BayesianNetworkLearner,
    EOSCModel,
    fit_eosc_cctv_benchmark_dag,
)

# Import metrics
from src.evaluation.metrics import (
    MetricsCollector,
    ActionOutcome,
    print_metrics_summary
)
from src.utils.bn_discretize import discretize_observation_like_training


class FaultInjector:
    """
    Simulates realistic fault scenarios in the Computing Continuum
    """
    
    @staticmethod
    def generate_normal_observation() -> Dict[str, Any]:
        """Generate observation for healthy system"""
        return {
            'cpu': random.uniform(30, 60),
            'memory': random.uniform(40, 70),
            'network': random.uniform(60, 100),
            'network_quality': 'high',
            'delay': random.uniform(10, 40),
            'fps': random.uniform(28, 30),
            'bitrate': random.uniform(2.5, 3.5),
            'streams': 3,
            'slo_violated': False,
            'device_type': 'edge'
        }
    
    @staticmethod
    def inject_network_drop(severity: float = 0.7) -> Dict[str, Any]:
        """
        Inject network bandwidth drop
        
        Args:
            severity: 0.0 to 1.0 (1.0 = severe)
        """
        # Network drops significantly
        network = random.uniform(10, 30) * (1 - severity)
        
        # This causes cascading effects:
        # - CPU increases (buffering, retransmissions)
        # - Delay increases
        # - FPS drops
        cpu = random.uniform(70, 95) * (1 + severity * 0.3)
        delay = random.uniform(80, 150) * (1 + severity * 0.5)
        fps = random.uniform(10, 20) * (1 - severity * 0.5)
        
        return {
            'cpu': min(100, cpu),
            'memory': random.uniform(50, 70),
            'network': max(5, network),
            'network_quality': 'low',
            'delay': delay,
            'fps': max(1, fps),
            'bitrate': random.uniform(0.5, 1.5),
            'streams': 3,
            'slo_violated': True,
            'device_type': 'edge'
        }
    
    @staticmethod
    def inject_cpu_spike(severity: float = 0.7) -> Dict[str, Any]:
        """
        Inject CPU spike (e.g., runaway process, malware)
        
        Args:
            severity: 0.0 to 1.0 (1.0 = severe)
        """
        # CPU spikes
        cpu = random.uniform(85, 99) * (1 + severity * 0.1)
        
        # This causes:
        # - Increased delay (processing bottleneck)
        # - Memory pressure
        # - FPS drops
        delay = random.uniform(100, 200) * (1 + severity * 0.3)
        memory = random.uniform(70, 90) * (1 + severity * 0.2)
        fps = random.uniform(15, 25) * (1 - severity * 0.3)
        
        return {
            'cpu': min(100, cpu),
            'memory': min(100, memory),
            'network': random.uniform(50, 80),
            'network_quality': 'medium',
            'delay': delay,
            'fps': max(5, fps),
            'bitrate': random.uniform(1.5, 2.5),
            'streams': 3,
            'slo_violated': True,
            'device_type': 'edge'
        }
    
    @staticmethod
    def inject_memory_leak(severity: float = 0.7) -> Dict[str, Any]:
        """
        Inject memory leak (gradual memory exhaustion)
        
        Args:
            severity: 0.0 to 1.0 (1.0 = severe)
        """
        # Memory increases over time
        memory = random.uniform(85, 98) * (1 + severity * 0.05)
        
        # This causes:
        # - CPU increase (GC thrashing)
        # - Delay increases (swapping)
        # - Eventual crash risk
        cpu = random.uniform(60, 85) * (1 + severity * 0.2)
        delay = random.uniform(60, 120) * (1 + severity * 0.4)
        
        return {
            'cpu': min(100, cpu),
            'memory': min(100, memory),
            'network': random.uniform(50, 80),
            'network_quality': 'medium',
            'delay': delay,
            'fps': random.uniform(20, 28),
            'bitrate': random.uniform(2.0, 3.0),
            'streams': 3,
            'slo_violated': True,
            'device_type': 'edge'
        }

    # ─── Synthetic network congestion (thesis fault injection) ─────────────────

    @staticmethod
    def inject_delay_spike(severity: float = 0.7) -> Dict[str, Any]:
        """
        Inject network delay spike (synthetic congestion: delay).
        """
        delay = random.uniform(80, 200) * (1 + severity * 0.3)
        cpu = random.uniform(60, 90)
        return {
            'cpu': min(100, cpu),
            'memory': random.uniform(40, 70),
            'network': random.uniform(40, 70),
            'network_quality': 'low',
            'delay': delay,
            'fps': random.uniform(15, 25),
            'bitrate': random.uniform(1.0, 2.0),
            'streams': 3,
            'slo_violated': True,
            'device_type': 'edge'
        }

    @staticmethod
    def inject_packet_loss(severity: float = 0.7) -> Dict[str, Any]:
        """
        Inject packet loss (synthetic congestion: packet loss).
        Modeled as low throughput and high delay.
        """
        throughput = random.uniform(0.3, 1.0) * (1 - severity * 0.3)
        delay = random.uniform(100, 180)
        return {
            'cpu': random.uniform(65, 92),
            'memory': random.uniform(50, 75),
            'network': max(5, throughput * 20),
            'network_quality': 'low',
            'delay': delay,
            'fps': random.uniform(10, 22),
            'bitrate': max(0.3, throughput),
            'streams': 3,
            'slo_violated': True,
            'device_type': 'edge'
        }

    @staticmethod
    def inject_overloaded_streams(severity: float = 0.7) -> Dict[str, Any]:
        """
        Inject overloaded streams (too many cameras/streams).
        """
        streams = random.randint(6, 12)
        cpu = random.uniform(80, 98) * (1 + severity * 0.1)
        delay = random.uniform(50, 120)
        fps = random.uniform(12, 25)
        return {
            'cpu': min(100, cpu),
            'memory': random.uniform(70, 92),
            'network': random.uniform(50, 80),
            'network_quality': 'medium',
            'delay': delay,
            'fps': fps,
            'bitrate': random.uniform(1.0, 2.0),
            'streams': streams,
            'slo_violated': True,
            'device_type': 'edge'
        }


class ExperimentRunner:
    """
    Runs controlled experiments comparing agents.

    AURORA-only options (defaults preserve prior behavior):

    - ``aurora_rca_mode``: ``"full"`` (ancestor RCA), ``"markov_blanket"`` (MB ablation),
      or ``"symphony"`` (full vs MB top-1 must agree).
    - ``aurora_online_cpt``: refit CCTV CPTs every ``aurora_online_cpt_every`` AURORA trials
      on initial training data plus a rolling buffer of trial observations; log max CPT
      drift via trial diagnostics ``online_cpt_delta``.
    """
    
    def __init__(
        self,
        num_trials: int = 50,
        output_dir: str = "experiments/results",
        *,
        aurora_rca_mode: str = "full",
        aurora_online_cpt: bool = False,
        aurora_online_cpt_every: int = 10,
        aurora_online_cpt_max_buffer: int = 400,
    ):
        self.num_trials = num_trials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.aurora_rca_mode = aurora_rca_mode
        self.aurora_online_cpt = aurora_online_cpt
        self.aurora_online_cpt_every = max(1, aurora_online_cpt_every)
        self.aurora_online_cpt_max_buffer = max(1, aurora_online_cpt_max_buffer)
        
        self.fault_types = {
            'network_drop': FaultInjector.inject_network_drop,
            'cpu_spike': FaultInjector.inject_cpu_spike,
            'memory_leak': FaultInjector.inject_memory_leak
        }
        
        # Ground truth: what is the actual root cause for each fault?
        self.ground_truth_root_causes = {
            'network_drop': 'network_quality',
            'cpu_spike': 'cpu',
            'memory_leak': 'memory'
        }
        
        # Ground truth: what is the correct action for each fault?
        self.correct_actions = {
            'network_drop': 'offload_to_fog',
            'cpu_spike': 'restart',
            'memory_leak': 'scale_up'
        }
        self.trial_diagnostics: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info(
            f"Experiment runner initialized: "
            f"{num_trials} trials × {len(self.fault_types)} faults × 3 agents = "
            f"{num_trials * len(self.fault_types) * 3} total trials"
        )
    
    def generate_training_data(self, num_samples: int = 1200) -> List[Dict[str, Any]]:
        """
        Generate synthetic training data for AIF agents
        
        This data teaches the causal relationships:
        - low network → high cpu → high delay → SLO violation
        - high cpu → high delay → SLO violation
        - high memory → high cpu → SLO violation
        """
        training_data = []
        profile_weights = [
            ("healthy", 0.30),
            ("network_drop", 0.25),
            ("memory_leak", 0.25),
            ("cpu_spike", 0.20),
        ]
        counts = {k: int(num_samples * w) for k, w in profile_weights}
        counts["healthy"] += num_samples - sum(counts.values())

        for profile, count in counts.items():
            for _ in range(count):
                if profile == "healthy":
                    network = random.uniform(55, 100)
                    memory = random.uniform(30, 70)
                elif profile == "network_drop":
                    network = random.uniform(8, 35)
                    memory = random.uniform(45, 75)
                elif profile == "memory_leak":
                    network = random.uniform(45, 80)
                    memory = random.uniform(82, 99)
                else:  # cpu_spike
                    network = random.uniform(35, 75)
                    memory = random.uniform(55, 88)

                # Stronger memory coupling (GC thrashing / swap pressure)
                mem_push = max(0.0, (memory - 68.0) * 0.95)
                cpu_base = 100 - network * 0.7
                cpu = cpu_base + random.uniform(-12, 12) + mem_push
                if profile == "cpu_spike":
                    cpu += random.uniform(10, 22)
                cpu = max(10, min(100, cpu))

                delay_base = cpu * 1.55 + mem_push * 0.45
                if profile == "network_drop":
                    delay_base += random.uniform(10, 40)
                delay = max(0, delay_base + random.uniform(-16, 16))

                slo_violated = (cpu > 80) or (network < 25) or (delay > 100) or (memory > 85)
                training_data.append({
                    'network_quality': 'low' if network < 40 else ('medium' if network < 70 else 'high'),
                    'cpu': 'high' if cpu > 70 else 'normal',
                    'memory': 'high' if memory > 75 else 'normal',
                    'delay': 'high' if delay > 80 else 'normal',
                    'slo_violated': slo_violated
                })
        
        return training_data

    def _derive_impact_calibration(self, rca: Any, training_data: List[Dict[str, Any]]) -> float:
        """Estimate impact calibration from run-specific impact distribution."""
        impacts: List[float] = []
        for row in training_data:
            if not row.get("slo_violated"):
                continue
            causes = rca.find_root_causes(
                symptom="slo_violated",
                symptom_value=True,
                observation=row,
                top_k=3,
            )
            impacts.extend([c.impact_score for c in causes if c.impact_score > 0])
            if len(impacts) >= 250:
                break
        if not impacts:
            return 0.42
        # 75th percentile gives "strong but not extreme" run-specific reference.
        return float(max(0.20, min(0.90, sorted(impacts)[int(0.75 * (len(impacts) - 1))])))
    
    def run_single_trial(
        self,
        trial_id: int,
        agent_wrapper: Any,
        agent_type: str,
        fault_type: str,
        collector: MetricsCollector
    ) -> None:
        """
        Run a single trial: inject fault, let agent respond, evaluate
        """
        # Get fault injector
        inject_fault = self.fault_types[fault_type]
        
        # Inject fault
        observation = inject_fault(severity=random.uniform(0.6, 0.9))
        
        # Start metrics collection
        collector.start_trial(
            trial_id=trial_id,
            agent_type=agent_type,
            fault_type=fault_type,
            initial_state=observation.copy()
        )
        
        # Agent processes observation
        start_time = time.time()
        result = agent_wrapper.process_observation(observation)
        end_time = time.time()
        
        # Extract result
        action_taken = result.get('action_taken')
        root_cause = result.get('root_cause')
        abstained = result.get('abstained', False)
        vfe = result.get('vfe')
        certainty = result.get('certainty')
        top_causes = result.get('top_causes', [])
        abstain_reason = result.get('abstain_reason', 'unspecified')

        # Determine outcome
        ground_truth_cause = self.ground_truth_root_causes[fault_type]
        correct_action = self.correct_actions[fault_type]
        
        if abstained:
            outcome = ActionOutcome.ABSTAINED
        elif action_taken is None:
            outcome = ActionOutcome.NO_ACTION
        elif action_taken == correct_action and root_cause == ground_truth_cause:
            outcome = ActionOutcome.CORRECT
        elif action_taken == correct_action:
            # Right action, but didn't identify root cause correctly
            outcome = ActionOutcome.CORRECT
        elif action_taken in ['restart', 'scale_up', 'offload_to_fog', 'reduce_load']:
            # Took an action, but wrong one
            # Could be destructive if it makes things worse
            if fault_type == 'network_drop' and action_taken == 'restart':
                outcome = ActionOutcome.DESTRUCTIVE  # Restarting won't fix network!
            elif fault_type == 'cpu_spike' and action_taken == 'scale_up':
                outcome = ActionOutcome.DESTRUCTIVE  # Scaling won't fix runaway process!
            else:
                outcome = ActionOutcome.INCORRECT
        else:
            outcome = ActionOutcome.INCORRECT
        
        # End metrics collection
        collector.end_trial(
            action_outcome=outcome,
            root_cause_identified=root_cause,
            action_taken=action_taken,
            final_state=observation.copy(),
            vfe_value=vfe,
            certainty_score=certainty,
            ground_truth_root_cause=ground_truth_cause
        )
        online_cpt_delta: Optional[float] = None
        if agent_type == "aurora" and hasattr(
            agent_wrapper, "after_trial_online_refresh"
        ):
            agent_wrapper.after_trial_online_refresh(observation)
            online_cpt_delta = getattr(
                agent_wrapper, "last_online_cpt_delta", None
            )
        self.trial_diagnostics.setdefault(agent_type, []).append({
            "trial_id": trial_id,
            "fault_type": fault_type,
            "ground_truth_root_cause": ground_truth_cause,
            "selected_root_cause": root_cause,
            "selected_action": action_taken,
            "certainty": certainty,
            "vfe": vfe,
            "outcome": outcome.value,
            "top3_causes": top_causes,
            "abstain_reason": abstain_reason,
            "vfe_pragmatic": result.get("vfe_pragmatic"),
            "vfe_epistemic": result.get("vfe_epistemic"),
            "ranking_margin": result.get("ranking_margin"),
            "top1_ranking_score": result.get("top1_ranking_score"),
            "top2_ranking_score": result.get("top2_ranking_score"),
            "symphony_views": result.get("symphony_views"),
            "aurora_rca_mode": result.get("aurora_rca_mode"),
            "online_cpt_delta": online_cpt_delta,
        })
        
        logger.info(
            f"Trial {trial_id} ({agent_type}/{fault_type}): "
            f"Action={action_taken}, Outcome={outcome.value}, "
            f"Time={(end_time - start_time)*1000:.1f}ms"
        )
    
    def run_experiment(self) -> Dict[str, Any]:
        """
        Run complete experiment comparing all agents
        
        Returns:
            Dictionary with results for each agent type
        """
        logger.info("="*70)
        logger.info("STARTING EXPERIMENTAL COMPARISON")
        logger.info("="*70)
        
        # Generate training data (shared by AIF agents)
        self.trial_diagnostics = {}
        training_data = self.generate_training_data(num_samples=1200)
        logger.info(f"Generated {len(training_data)} training samples")
        
        # Shared BN: fixed DAG so memory/network/cpu are all admissible root causes
        shared_eosc = fit_eosc_cctv_benchmark_dag(training_data)
        
        # Initialize agents
        agents = {
            'rule_based': RuleBasedAgentWrapper("rule-baseline"),
            'aif_no_gate': AIFNoGateAgentWrapper("aif-no-gate-baseline"),
            'aurora': self._create_aurora_agent(
                training_data,
                eosc_model=shared_eosc,
            ),
        }
        
        # Train AIF agents
        logger.info("Training AIF-based agents...")
        agents['aif_no_gate'].train(training_data, eosc_model=shared_eosc)
        logger.success("All agents ready")
        
        # Run experiments for each agent
        results = {}
        
        for agent_name, agent_wrapper in agents.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"TESTING: {agent_name.upper()}")
            logger.info(f"{'='*70}")
            
            collector = MetricsCollector()
            trial_counter = 0
            
            # Run trials for each fault type
            for fault_type in self.fault_types.keys():
                logger.info(f"\nFault type: {fault_type}")
                
                for trial_num in range(self.num_trials):
                    trial_counter += 1
                    self.run_single_trial(
                        trial_id=trial_counter,
                        agent_wrapper=agent_wrapper,
                        agent_type=agent_name,
                        fault_type=fault_type,
                        collector=collector,
                    )
                    
                    # Small delay between trials
                    time.sleep(0.1)
            
            # Save results for this agent
            output_file = self.output_dir / f"{agent_name}_results.json"
            collector.save_results(str(output_file))
            with open(self.output_dir / f"{agent_name}_trial_diagnostics.json", "w") as f:
                json.dump(self.trial_diagnostics.get(agent_name, []), f, indent=2)
            
            # Compute aggregated metrics
            aggregated = collector.get_aggregated_metrics()
            results[agent_name] = aggregated
            
            # Print summary
            print_metrics_summary(aggregated)
        
        # Save comparison summary
        self._save_comparison_summary(results)
        
        logger.success("Experiment completed!")
        return results
    
    def _create_aurora_agent(
        self,
        training_data: List[Dict[str, Any]],
        eosc_model: EOSCModel,
        certainty_threshold: float = 0.70,
        vfe_threshold: float = 3.85,
        impact_calibration: Optional[float] = None,
        ranking_margin_threshold: float = 0.04,
    ) -> Any:
        """
        Create and train AURORA agent (certainty, ranking margin, VFE threshold).

        For experiments, we create a simplified wrapper that uses the core
        AURORA logic without needing to instantiate all micro-agents.

        ``ranking_margin_threshold``: abstain with ``abstain_reason='ambiguous'`` when
        top-1 and top-2 ``RootCause.ranking_score`` differ by less than this (on the
        same scale as the RCA composite score, typically O(0.01--0.2)).

        Runner-level ``aurora_rca_mode``: ``full`` (default), ``markov_blanket`` (MB ablation),
        or ``symphony`` (dual-view consensus; abstain if full vs MB top-1 disagree).

        Runner-level ``aurora_online_cpt``: periodically refit CCTV CPTs on base training
        data plus a buffer of trial observations (see ``max_abs_cpd_delta`` in logs).
        """
        from src.aif.root_cause_analyzer import RootCauseAnalyzer
        from src.aif.vfe_compute import VFEComputer, UncertaintyGate
        from src.aif import two_stage_gate as tsg

        bn_learner = BayesianNetworkLearner()
        eosc_model = bn_learner.update_parameters(eosc_model, training_data)
        
        # Create core components
        rca = RootCauseAnalyzer(eosc_model)
        vfe_computer = VFEComputer()
        gate = UncertaintyGate(threshold=vfe_threshold)
        if impact_calibration is None:
            impact_calibration = self._derive_impact_calibration(rca, training_data)
        
        rca_mode = getattr(self, "aurora_rca_mode", "full")
        online_cpt = getattr(self, "aurora_online_cpt", False)
        online_every = getattr(self, "aurora_online_cpt_every", 10)
        online_cap = getattr(self, "aurora_online_cpt_max_buffer", 400)

        # Wrapper that implements AURORA logic
        class AURORAWrapper:
            def __init__(self, eosc_model, rca, vfe_computer, gate):
                self.eosc_model = eosc_model
                self.rca = rca
                self.vfe_computer = vfe_computer
                self.gate = gate
                self.agent_type = "aurora"
                self.certainty_threshold = certainty_threshold
                self.impact_calibration = impact_calibration
                self.ranking_margin_threshold = ranking_margin_threshold
                self.rca_mode = rca_mode
                self.training_data_base = list(training_data)
                self.online_cpt = online_cpt
                self.online_cpt_every = online_every
                self.online_cpt_max_buffer = online_cap
                self._online_rows: List[Dict[str, Any]] = []
                self._aurora_trial_seq = 0
                self.last_online_cpt_delta: Optional[float] = None

            def after_trial_online_refresh(self, observation: Dict[str, Any]) -> None:
                """Refit fixed-DAG CPTs on base + rolling buffer (AURORA only)."""
                if not self.online_cpt:
                    return
                from src.aif.bayesian_network import (
                    max_abs_cpd_delta,
                    refit_cctv_benchmark_cpts,
                )

                self._aurora_trial_seq += 1
                row = discretize_observation_like_training(observation)
                row["slo_violated"] = bool(observation.get("slo_violated"))
                self._online_rows.append(row)
                if len(self._online_rows) > self.online_cpt_max_buffer:
                    self._online_rows = self._online_rows[-self.online_cpt_max_buffer :]
                if self._aurora_trial_seq % self.online_cpt_every != 0:
                    return
                prev = self.eosc_model
                merged = self.training_data_base + self._online_rows
                new_e = refit_cctv_benchmark_cpts(merged)
                self.last_online_cpt_delta = max_abs_cpd_delta(prev, new_e)
                self.eosc_model = new_e
                self.rca = RootCauseAnalyzer(self.eosc_model)
            
            def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
                """
                Two-stage AURORA for experiments:
                1) diagnose (RCA + certainty + candidate action)
                2) VFE on action-conditioned BN hypothesis → gate
                """
                slo_violated = observation.get('slo_violated', False)

                if not slo_violated:
                    return {
                        'action_taken': None,
                        'root_cause': None,
                        'abstained': False,
                        'vfe': 0.0,
                        'certainty': 1.0,
                        'abstain_reason': tsg.ABSTAIN_NO_SLO,
                        'top_causes': [],
                    }

                obs_bn = discretize_observation_like_training(observation)

                if self.rca_mode == "symphony":
                    dx = tsg.diagnose_slo_violation_symphony(
                        self.rca,
                        obs_bn,
                        top_k=3,
                        impact_calibration=self.impact_calibration,
                    )
                elif self.rca_mode == "markov_blanket":
                    dx = tsg.diagnose_slo_violation(
                        self.rca,
                        obs_bn,
                        top_k=3,
                        impact_calibration=self.impact_calibration,
                        use_markov_blanket=True,
                    )
                else:
                    dx = tsg.diagnose_slo_violation(
                        self.rca,
                        obs_bn,
                        top_k=3,
                        impact_calibration=self.impact_calibration,
                        use_markov_blanket=False,
                    )
                root_causes = dx["root_causes"]
                certainty = dx["certainty"]
                candidate_action = dx["candidate_action"]
                top_summaries = dx["top_summaries"]

                if not root_causes:
                    return {
                        'action_taken': None,
                        'root_cause': None,
                        'abstained': True,
                        'vfe': 0.0,
                        'certainty': 0.0,
                        'abstain_reason': tsg.ABSTAIN_NO_CAUSES,
                        'top_causes': [],
                    }

                top_cause = dx["top_cause"]
                rank_margin = dx.get("ranking_margin")
                top1_rs = dx.get("top1_ranking_score")
                top2_rs = dx.get("top2_ranking_score")

                def _payload(
                    vfe: float,
                    vp: Optional[float],
                    ve: Optional[float],
                    abstain_reason: str,
                    acted: bool,
                    action_name: Optional[str],
                ) -> Dict[str, Any]:
                    return {
                        'action_taken': action_name,
                        'root_cause': top_cause.variable,
                        'abstained': not acted,
                        'vfe': vfe,
                        'certainty': certainty,
                        'abstain_reason': abstain_reason,
                        'top_causes': top_summaries,
                        'vfe_pragmatic': vp,
                        'vfe_epistemic': ve,
                        'ranking_margin': rank_margin,
                        'top1_ranking_score': top1_rs,
                        'top2_ranking_score': top2_rs,
                        'symphony_views': dx.get("symphony_views"),
                        'symphony_disagree': dx.get("symphony_disagree", False),
                        'aurora_rca_mode': self.rca_mode,
                    }

                try:
                    dec = self.vfe_computer.compute_vfe_decomposed(
                        model=self.eosc_model,
                        evidence=obs_bn,
                        action=candidate_action,
                    )
                    vfe = dec["vfe"]
                    vfp, vfe_ep = dec["pragmatic"], dec["epistemic"]
                except Exception:
                    vfe, vfp, vfe_ep = 1.5, None, None

                if dx.get("symphony_disagree"):
                    return _payload(
                        vfe,
                        vfp,
                        vfe_ep,
                        tsg.ABSTAIN_SYMPHONY,
                        False,
                        None,
                    )

                # Dynamic margin logic for AURORA in experiments
                criticality_score = 1
                if observation.get('cpu', 0) > 90:
                    criticality_score += 1
                if observation.get('memory', 0) > 90:
                    criticality_score += 1
                if observation.get('network', 100) < 20:
                    criticality_score += 1

                effective_margin = max(0.02, self.ranking_margin_threshold - 0.04 * (criticality_score - 1))

                if tsg.is_ambiguous_ranking_margin(
                    root_causes, effective_margin
                ):
                    return _payload(
                        vfe,
                        vfp,
                        vfe_ep,
                        tsg.ABSTAIN_AMBIGUOUS_RANKING,
                        False,
                        None,
                    )

                # Dynamic Threshold Logic for AURORA in experiments
                dynamic_threshold = self.certainty_threshold
                criticality_score = 1
                if observation.get('cpu', 0) > 90:
                    criticality_score += 1
                if observation.get('memory', 0) > 90:
                    criticality_score += 1
                if observation.get('network', 100) < 20:
                    criticality_score += 1

                if criticality_score > 1:
                    dynamic_threshold = max(0.50, self.certainty_threshold - (0.15 * min(criticality_score - 1, 2)))

                if certainty < dynamic_threshold:
                    return _payload(
                        vfe, vfp, vfe_ep, tsg.ABSTAIN_LOW_CERTAINTY, False, None
                    )

                if not tsg.should_execute_from_vfe(vfe, self.gate):
                    return _payload(
                        vfe, vfp, vfe_ep, tsg.ABSTAIN_HIGH_VFE, False, None
                    )

                return _payload(
                    vfe, vfp, vfe_ep, tsg.ABSTAIN_NONE, True, candidate_action
                )
            
            def reset(self):
                pass
        
        return AURORAWrapper(eosc_model, rca, vfe_computer, gate)

    def run_threshold_sweep(
        self,
        certainty_thresholds: List[float],
        vfe_thresholds: List[float],
    ) -> List[Dict[str, Any]]:
        """
        Run AURORA threshold sweep and save summary for Pareto analysis.
        """
        sweep_rows: List[Dict[str, Any]] = []
        training_data = self.generate_training_data(num_samples=1200)
        shared_eosc = fit_eosc_cctv_benchmark_dag(training_data)

        for c_thr in certainty_thresholds:
            for v_thr in vfe_thresholds:
                collector = MetricsCollector()
                agent = self._create_aurora_agent(
                    training_data=training_data,
                    eosc_model=shared_eosc,
                    certainty_threshold=c_thr,
                    vfe_threshold=v_thr,
                )
                trial_counter = 0
                self.trial_diagnostics["aurora_sweep"] = []
                for fault_type in self.fault_types.keys():
                    for _ in range(self.num_trials):
                        trial_counter += 1
                        self.run_single_trial(
                            trial_id=trial_counter,
                            agent_wrapper=agent,
                            agent_type="aurora_sweep",
                            fault_type=fault_type,
                            collector=collector,
                        )
                agg = collector.get_aggregated_metrics()
                sweep_rows.append({
                    "certainty_threshold": c_thr,
                    "vfe_threshold": v_thr,
                    "repair_accuracy": agg.mean_repair_accuracy,
                    "abstention_rate": agg.abstention_rate,
                    "destructive_action_rate": agg.destructive_action_rate,
                    "resolution_rate": agg.resolution_rate,
                })

        with open(self.output_dir / "threshold_sweep_summary.json", "w") as f:
            json.dump(sweep_rows, f, indent=2)
        return sweep_rows
    
    def _save_comparison_summary(self, results: Dict[str, Any]):
        """Save a comparison summary to JSON"""
        summary = {}
        
        for agent_name, aggregated in results.items():
            summary[agent_name] = aggregated.to_dict()
        
        output_file = self.output_dir / "comparison_summary.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Comparison summary saved to {output_file}")


# Main execution (50-100 trials for thesis; default 50 per fault type)
if __name__ == "__main__":
    runner = ExperimentRunner(num_trials=150, output_dir="experiments/results")
    results = runner.run_experiment()
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print("\nResults saved to: experiments/results/")
    print("  • rule_based_results.json")
    print("  • aif_no_gate_results.json")
    print("  • aurora_results.json")
    print("  • comparison_summary.json")

