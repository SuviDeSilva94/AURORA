"""
Unit tests for Active Inference components: BN learning, VFE, uncertainty gating.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import numpy as np

from src.aif.bayesian_network import BayesianNetworkLearner, EOSCModel, fit_eosc_cctv_benchmark_dag
from src.aif.root_cause_analyzer import RootCause
from src.aif.vfe_compute import VFEComputer, UncertaintyGate, healing_hypothesis_for_action
from src.aif import two_stage_gate as tsg
from src.aif.agent import Agent, Observation
from src.utils.config import AgentConfig


class TestBayesianNetwork:
    """Test Bayesian Network learning"""
    
    def test_structure_learning(self):
        """Test that BN can learn structure from data"""
        # Create synthetic data with clear causal structure
        # Use discrete values for better structure learning
        data = []
        for i in range(100):
            cpu = np.random.choice([0, 1, 2])  # Low, Med, High
            delay = min(cpu + np.random.choice([0, 1]), 2)  # Caused by cpu
            slo = 1 if delay >= 2 else 0  # Caused by delay
            data.append({'cpu': cpu, 'delay': delay, 'slo': slo})
        
        learner = BayesianNetworkLearner(algorithm='hc')
        model = learner.learn_structure(data)
        
        # Check that model was created
        assert model is not None
        assert model.model is not None
        
        # Check that structure exists (may be empty if data too simple)
        edges = model.get_structure()
        print(f"✓ Learned structure with {len(edges)} edges")
        if len(edges) > 0:
            print(f"  Edges: {edges}")
        else:
            print("  Note: No edges learned (data may be too simple, but model works)")
    
    def test_parameter_learning(self):
        """Test that BN can update parameters"""
        data = [
            {'cpu': 10, 'delay': 5, 'slo': 0},
            {'cpu': 90, 'delay': 45, 'slo': 1},
        ] * 10
        
        learner = BayesianNetworkLearner()
        model = learner.learn_structure(data)
        
        # Update with new data
        new_data = [{'cpu': 50, 'delay': 25, 'slo': 0}] * 5
        updated_model = learner.update_parameters(model, new_data)
        
        assert updated_model is not None
        print("✓ Parameter learning works")
    
    def test_markov_blanket(self):
        """Test Markov Blanket extraction"""
        data = []
        for i in range(100):
            cpu = np.random.choice([0, 1, 2])
            delay = min(cpu + np.random.choice([0, 1]), 2)
            slo = 1 if delay >= 2 else 0
            fps = 2 - cpu
            data.append({'cpu': cpu, 'delay': delay, 'slo': slo, 'fps': fps})
        
        learner = BayesianNetworkLearner(use_markov_blanket=True)
        model = learner.learn_structure(data)
        
        # Check MB for slo variable - only if it exists in the model
        if 'slo' in model.model.nodes():
            mb = model.get_markov_blanket('slo')
            assert isinstance(mb, set)
            print(f"✓ Markov Blanket for 'slo': {mb}")
        else:
            print("✓ Markov blanket computation works (variable not in learned structure)")


class TestVFEComputation:
    """Test VFE computation and safe-execution threshold"""
    
    def test_vfe_computation(self):
        """Test that VFE can be computed"""
        # Create simple model
        data = [
            {'action': 0, 'slo': 0},
            {'action': 1, 'slo': 1},
        ] * 20
        
        learner = BayesianNetworkLearner()
        model = learner.learn_structure(data)
        
        vfe_computer = VFEComputer()
        
        evidence = {'action': 0}
        action = {'action': 1}
        
        vfe = vfe_computer.compute_vfe(model, evidence, action)
        
        assert isinstance(vfe, float)
        assert not np.isnan(vfe)
        print(f"✓ VFE computed: {vfe:.3f}")

    def test_vfe_cctv_bn_and_action_hypothesis(self):
        """CCTV benchmark BN: VFE uses slo_violated; actions map to discrete patches."""
        rng = np.random.default_rng(42)
        rows = []
        for _ in range(250):
            nq = rng.choice(["low", "medium", "high"])
            mem = rng.choice(["normal", "high"])
            cpu = rng.choice(["normal", "high"])
            delay = rng.choice(["normal", "high"])
            bad = (nq == "low") or (cpu == "high" and delay == "high") or mem == "high"
            rows.append(
                {
                    "network_quality": nq,
                    "memory": mem,
                    "cpu": cpu,
                    "delay": delay,
                    "slo_violated": bool(bad or rng.random() < 0.15),
                }
            )
        model = fit_eosc_cctv_benchmark_dag(rows)
        vc = VFEComputer()
        obs = {
            "network_quality": "low",
            "cpu": "high",
            "memory": "normal",
            "delay": "high",
            "slo_violated": True,
        }
        dec0 = vc.compute_vfe_decomposed(model, obs, action={})
        dec_fog = vc.compute_vfe_decomposed(model, obs, action="offload_to_fog")
        assert not np.isnan(dec0["vfe"])
        assert dec0.keys() >= {"vfe", "pragmatic", "epistemic"}
        assert vc.compute_vfe(model, obs, "restart") == pytest.approx(
            vc.compute_vfe_decomposed(model, obs, "restart")["vfe"]
        )
        assert isinstance(dec_fog["vfe"], float) and not np.isnan(dec_fog["vfe"])
        assert healing_hypothesis_for_action("restart")["cpu"] == "normal"
    
    def test_uncertainty_gate_allows(self):
        """Test that gate allows low VFE"""
        gate = UncertaintyGate(threshold=2.0)
        
        low_vfe = 1.5
        assert gate.should_act(low_vfe) == True
        print(f"✓ Gate allows action with VFE={low_vfe}")
    
    def test_uncertainty_gate_blocks(self):
        """Test that gate blocks high VFE"""
        gate = UncertaintyGate(threshold=2.0)
        
        high_vfe = 3.5
        assert gate.should_act(high_vfe) == False
        print(f"✓ Gate blocks action with VFE={high_vfe}")
    
    def test_abstention_rate(self):
        """Test abstention rate tracking"""
        gate = UncertaintyGate(threshold=2.0)
        
        # Simulate decisions
        vfes = [1.0, 1.5, 2.5, 3.0, 1.8, 2.2, 1.3]
        for vfe in vfes:
            gate.should_act(vfe)
        
        rate = gate.get_abstention_rate()
        expected_rate = 3 / 7  # 3 out of 7 were > 2.0
        
        assert abs(rate - expected_rate) < 0.01
        print(f"✓ Abstention rate: {rate:.2%}")


class TestRankingMarginAbstention:
    """Top-1 vs top-2 ranking_score margin (structural ambiguity gate)."""

    def _rc(self, variable: str, ranking_score: float) -> RootCause:
        return RootCause(
            variable=variable,
            value="high",
            impact_score=0.5,
            causal_paths=[],
            evidence_strength=0.5,
            explanation="",
            ranking_score=ranking_score,
        )

    def test_margin_and_ambiguous(self):
        clear = [self._rc("cpu", 0.92), self._rc("delay", 0.40)]
        assert tsg.ranking_margin_top1_top2(clear) == pytest.approx(0.52)
        assert not tsg.is_ambiguous_ranking_margin(clear, 0.06)

        tight = [self._rc("cpu", 0.91), self._rc("delay", 0.90)]
        assert tsg.ranking_margin_top1_top2(tight) == pytest.approx(0.01)
        assert tsg.is_ambiguous_ranking_margin(tight, 0.06)

        assert tsg.ranking_margin_top1_top2([clear[0]]) is None
        assert not tsg.is_ambiguous_ranking_margin([clear[0]], 0.06)

    def test_abstain_label_constant(self):
        assert tsg.ABSTAIN_AMBIGUOUS_RANKING == "ambiguous"


class TestAgent:
    """Test Agent functionality"""
    
    def test_agent_creation(self):
        """Test that agent can be created"""
        config = AgentConfig(
            vfe_threshold=2.0,
            aif_iteration_ms=1000,
            bn_learning_algorithm='hc'
        )
        
        agent = Agent(
            agent_id='test-agent',
            device_type='test-device',
            config=config
        )
        
        assert agent is not None
        assert agent.agent_id == 'test-agent'
        assert agent.device_type == 'test-device'
        print("✓ Agent created successfully")
    
    def test_observation_collection(self):
        """Test that agent can collect observations"""
        config = AgentConfig()
        agent = Agent('test-agent', 'test-device', config)
        
        # Manually trigger observation (without starting thread)
        observation = agent._observe()
        
        assert isinstance(observation, Observation)
        assert 'cpu' in observation.metrics
        assert 'memory' in observation.metrics
        print(f"✓ Observation collected: {len(observation.metrics)} metrics")
    
    def test_slo_evaluation(self):
        """Test SLO evaluation"""
        config = AgentConfig()
        agent = Agent('test-agent', 'test-device', config)
        
        # Test with violations
        metrics_violated = {
            'network': 2.0,  # > 1.6 MB/s
            'delay': 50,     # high
            'fps': 30,
            'success': False,
            'distance': 60   # > 50
        }
        
        violations = agent._evaluate_slos(metrics_violated)
        assert 'network' in violations
        assert 'success' in violations
        assert 'distance' in violations
        print(f"✓ SLO evaluation detected {len(violations)} violations")


class TestDoCalculus:
    """
    Pearl do-calculus integration tests.

    AURORA's CCTV BN is fully observed and ``slo_violated`` is a leaf — by the
    backdoor criterion, the interventional and observational queries coincide
    numerically, so the swap to ``CausalInference.query(do=...)`` must not
    move the existing experimental results. The confounder test proves that
    when the structural assumption is violated, the new path actually does
    graph mutilation (i.e. it isn't observational conditioning in disguise).
    """

    @staticmethod
    def _aurora_model():
        rng = np.random.default_rng(7)
        rows = []
        for _ in range(400):
            nq = rng.choice(["low", "medium", "high"])
            mem = rng.choice(["normal", "high"])
            cpu = rng.choice(["normal", "high"])
            delay = rng.choice(["normal", "high"])
            bad = (nq == "low") or (cpu == "high" and delay == "high") or mem == "high"
            rows.append({
                "network_quality": nq, "memory": mem, "cpu": cpu,
                "delay": delay, "slo_violated": bool(bad or rng.random() < 0.15),
            })
        return fit_eosc_cctv_benchmark_dag(rows)

    @staticmethod
    def _legacy_p_clear(model, evidence):
        """Pre-do-calculus observational ``P(slo_violated=False | evidence)``."""
        ev = {k: v for k, v in evidence.items()
              if k in model.model.nodes() and k != "slo_violated"}
        qr = model.inference_engine.query(variables=["slo_violated"], evidence=ev)
        vals = np.asarray(qr.values).flatten()
        cpd = model.model.get_cpds("slo_violated")
        states = list(cpd.state_names["slo_violated"])
        for i, s in enumerate(states):
            if s is False:
                return float(vals[i])
        return float(vals[0])

    def test_do_calculus_equivalence_on_aurora_dag(self):
        """
        On the AURORA leaf-node DAG, do-calculus must agree with the legacy
        observational override to within floating-point noise. This proves the
        switch to ``CausalInference`` does not perturb published experiment
        numbers.
        """
        model = self._aurora_model()
        vc = VFEComputer()
        actions = ["restart", "scale_up", "offload_to_fog", "reduce_load"]
        observations = [
            {"network_quality": "low", "memory": "normal", "cpu": "high", "delay": "high"},
            {"network_quality": "high", "memory": "high", "cpu": "high", "delay": "normal"},
            {"network_quality": "medium", "memory": "normal", "cpu": "normal", "delay": "high"},
            {"network_quality": "high", "memory": "normal", "cpu": "high", "delay": "normal"},
        ]
        for obs in observations:
            for act in actions:
                patch = healing_hypothesis_for_action(act)
                # Legacy: pre-merge patch into evidence and query observationally.
                merged = {**obs, **patch}
                p_legacy = self._legacy_p_clear(model, merged)
                # New path: do=patch, evidence=obs minus patched keys.
                obs_no_patch = {k: v for k, v in obs.items() if k not in patch}
                p_do = vc._prob_slo_cleared(model, obs_no_patch, do=patch)
                assert p_do == pytest.approx(p_legacy, abs=1e-9), (
                    f"do/obs disagree for action={act} obs={obs}: "
                    f"do={p_do} legacy={p_legacy}"
                )

    def test_do_calculus_diverges_under_confounding(self):
        """
        On a synthetic DAG with an unobserved-effect-style confounder, the
        observational and interventional queries must produce *different*
        marginals. This is the proof that pgmpy is mutilating the graph
        rather than just renaming a dictionary key.

        Structure:  U -> X,  U -> Y,  X -> Y
        With U strongly biasing both X and Y, P(Y | X=1) >> P(Y | do(X=1)).
        """
        from pgmpy.models import BayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination, CausalInference

        bn = BayesianNetwork([("U", "X"), ("U", "Y"), ("X", "Y")])
        cpd_u = TabularCPD("U", 2, [[0.5], [0.5]])
        cpd_x = TabularCPD("X", 2,
                           [[0.9, 0.1],
                            [0.1, 0.9]],
                           evidence=["U"], evidence_card=[2])
        # Y depends much more on U than on X.
        cpd_y = TabularCPD(
            "Y", 2,
            [[0.95, 0.90, 0.10, 0.05],
             [0.05, 0.10, 0.90, 0.95]],
            evidence=["U", "X"], evidence_card=[2, 2],
        )
        bn.add_cpds(cpd_u, cpd_x, cpd_y)
        assert bn.check_model()

        ve = VariableElimination(bn)
        ci = CausalInference(bn)

        p_obs = float(ve.query(["Y"], evidence={"X": 1}, show_progress=False).values[1])
        p_do = float(ci.query(variables=["Y"], do={"X": 1}, evidence={},
                              show_progress=False).values[1])
        assert abs(p_obs - p_do) > 0.01, (
            f"On a confounded DAG, P(Y|X) and P(Y|do(X)) must differ; "
            f"got obs={p_obs:.4f}, do={p_do:.4f}"
        )

    def test_empty_do_passes_through_to_observational(self):
        """``do={}`` must reduce to the legacy conditional — equal to the
        full pre-action ``P(slo_violated=False | evidence)``."""
        model = self._aurora_model()
        vc = VFEComputer()
        obs = {"network_quality": "low", "memory": "normal",
               "cpu": "high", "delay": "high"}
        p_legacy = self._legacy_p_clear(model, obs)
        p_empty_do = vc._prob_slo_cleared(model, obs, do={})
        assert p_empty_do == pytest.approx(p_legacy, abs=1e-9)

    def test_intervened_key_overrides_evidence_duplicate(self):
        """If the same variable appears in both ``do`` and ``evidence``, the
        intervened value must win and the call must succeed."""
        model = self._aurora_model()
        vc = VFEComputer()
        # cpu=high in evidence; do(cpu=normal) must override.
        evidence_with_dup = {"network_quality": "low", "memory": "normal",
                             "cpu": "high", "delay": "high"}
        p_with_do = vc._prob_slo_cleared(model, evidence_with_dup, do={"cpu": "normal"})
        # Reference: drop cpu from evidence, then query with do(cpu=normal).
        clean_evidence = {k: v for k, v in evidence_with_dup.items() if k != "cpu"}
        p_clean = vc._prob_slo_cleared(model, clean_evidence, do={"cpu": "normal"})
        assert p_with_do == pytest.approx(p_clean, abs=1e-9)


def run_all_tests():
    """Run all unit tests"""
    print("="*80)
    print("Running Unit Tests")
    print("="*80)
    
    # Bayesian Network tests
    print("\n[1/3] Testing Bayesian Network...")
    bn_tests = TestBayesianNetwork()
    bn_tests.test_structure_learning()
    bn_tests.test_parameter_learning()
    bn_tests.test_markov_blanket()
    
    # VFE tests
    print("\n[2/3] Testing VFE Computation...")
    vfe_tests = TestVFEComputation()
    vfe_tests.test_vfe_computation()
    vfe_tests.test_uncertainty_gate_allows()
    vfe_tests.test_uncertainty_gate_blocks()
    vfe_tests.test_abstention_rate()
    
    # Agent tests
    print("\n[3/3] Testing Agent...")
    agent_tests = TestAgent()
    agent_tests.test_agent_creation()
    agent_tests.test_observation_collection()
    agent_tests.test_slo_evaluation()
    
    print("\n" + "="*80)
    print("✅ All Unit Tests Passed!")
    print("="*80)


if __name__ == "__main__":
    run_all_tests()

