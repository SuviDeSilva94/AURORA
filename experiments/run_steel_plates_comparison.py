"""
Second real-data evaluation: Steel Plates Faults Dataset (UCI).

Third-party industrial fault dataset from UCI: visual surface defects in
steel plates classified into 7 defect categories. We use three classes
that are most statistically distinct on the available geometric features:

  K_Scatch    n=391   large surface defects, negative orientation
  Stains      n=72    very small high-edge defects
  Pastry      n=158   medium-sized irregular shape

Each row carries 27 continuous geometric/photometric features. The
evaluation discretises four of these (pixels_area, log_area, orientation,
edges_index) into BN states, fits a Bayesian network, and runs the same
three-agent comparison used on CCTV, motor, and AI4I 2020.

Sandbox: writes JSONs under experiments/results/steel_*_results.json.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
import numpy as np
from loguru import logger
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork

from src.aif.bayesian_network import EOSCModel, BayesianNetworkLearner
from src.aif.root_cause_analyzer import RootCauseAnalyzer
from src.aif.two_stage_gate import (
    dynamic_tau_from_anomalies,
    diagnose_slo_violation,
)
from src.aif.vfe_compute import VFEComputer, UncertaintyGate
import src.aif.two_stage_gate as tsg
import src.aif.vfe_compute as vfe_mod


DATASET_PATH = Path("/Users/suvidesilva/Downloads/ai_agnt/experiments/data/steel_plates_faults.dat")
DATASET_COLS = [
    "X_Minimum","X_Maximum","Y_Minimum","Y_Maximum","Pixels_Areas","X_Perimeter","Y_Perimeter",
    "Sum_of_Luminosity","Minimum_of_Luminosity","Maximum_of_Luminosity","Length_of_Conveyer",
    "TypeOfSteel_A300","TypeOfSteel_A400","Steel_Plate_Thickness","Edges_Index","Empty_Index",
    "Square_Index","Outside_X_Index","Edges_X_Index","Edges_Y_Index","Outside_Global_Index",
    "LogOfAreas","Log_X_Index","Log_Y_Index","Orientation_Index","Luminosity_Index","SigmoidOfAreas",
    "Pastry","Z_Scratch","K_Scatch","Stains","Dirtiness","Bumps","Other_Faults",
]

FAULT_CLASSES = ["K_Scatch", "Stains", "Pastry"]


# ─── Discretisation ─────────────────────────────────────────────────────────

def discretize_observation(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map continuous defect features to discrete BN labels.

    Thresholds derived from empirical distribution per fault class:
      K_Scatch:  Pixels_Areas ~7600 (huge), LogOfAreas ~3.6, Orientation ~-0.3
      Stains:    Pixels_Areas ~20  (tiny), Edges_Index ~0.58
      Pastry:    Pixels_Areas ~560, Orientation ~+0.64
    """
    pa = float(row.get("Pixels_Areas", 0))
    la = float(row.get("LogOfAreas", 0))
    oi = float(row.get("Orientation_Index", 0))
    ei = float(row.get("Edges_Index", 0))

    out = {
        "pixels_area": "large" if pa > 1000 else ("small" if pa < 50 else "normal"),
        "log_area":    "high" if la > 3.0 else "normal",
        "orientation": "positive" if oi > 0.3 else ("negative" if oi < -0.2 else "neutral"),
        "edges_idx":   "high" if ei > 0.45 else ("low" if ei < 0.2 else "normal"),
    }
    if "slo_violated" in row:
        out["slo_violated"] = row["slo_violated"]
    return out


def observed_abnormal_vars(row: Dict[str, Any]) -> Set[str]:
    """Which variables are observed in abnormal state on this trial?"""
    pa = float(row.get("Pixels_Areas", 0))
    la = float(row.get("LogOfAreas", 0))
    oi = float(row.get("Orientation_Index", 0))
    ei = float(row.get("Edges_Index", 0))
    abnormal = set()
    if pa > 1000 or pa < 50: abnormal.add("pixels_area")
    if la > 3.0: abnormal.add("log_area")
    if oi > 0.3 or oi < -0.2: abnormal.add("orientation")
    if ei > 0.45 or ei < 0.2: abnormal.add("edges_idx")
    return abnormal


def count_anomalies(obs: Dict[str, Any]) -> int:
    return max(1, len(observed_abnormal_vars(obs)))


# ─── Dataset loading ───────────────────────────────────────────────────────

def load_dataset():
    df = pd.read_csv(DATASET_PATH, sep="\t", header=None, names=DATASET_COLS)

    def label_fault(r):
        for f in FAULT_CLASSES:
            if r[f] == 1:
                return f
        return None

    df["fault_label"] = df.apply(label_fault, axis=1)
    fault_trials = df[df["fault_label"].notna()].copy()
    healthy = df[df[FAULT_CLASSES + ["Z_Scratch", "Dirtiness", "Bumps", "Other_Faults"]].sum(axis=1) == 0]
    # No truly healthy rows in this dataset (every plate has SOME defect by definition)
    # Use the OTHER fault types as "control" for BN training
    healthy_proxy = df[~df["fault_label"].notna()].sample(n=min(800, len(df) - len(fault_trials)), random_state=42)
    training_rows = pd.concat([healthy_proxy, fault_trials], ignore_index=True)
    return training_rows, fault_trials


# ─── BN structure ──────────────────────────────────────────────────────────

STEEL_EDGES = [
    ("pixels_area", "quality_status"),
    ("log_area", "quality_status"),
    ("orientation", "quality_status"),
    ("edges_idx", "quality_status"),
    ("quality_status", "slo_violated"),
    ("pixels_area", "slo_violated"),
    ("log_area", "slo_violated"),
    ("orientation", "slo_violated"),
    ("edges_idx", "slo_violated"),
]


def fit_steel_bn(training_df: pd.DataFrame) -> EOSCModel:
    rows = []
    for _, r in training_df.iterrows():
        d = discretize_observation(r)
        # Latent quality_status: derive from fault label
        if r.get("fault_label") is None:
            d["quality_status"] = "good"
            d["slo_violated"] = False
        else:
            d["quality_status"] = "defective"
            d["slo_violated"] = True
        rows.append(d)
    bn_df = pd.DataFrame(rows)
    bn_df = bn_df[["pixels_area", "log_area", "orientation", "edges_idx",
                    "quality_status", "slo_violated"]]
    model = BayesianNetwork(STEEL_EDGES)
    model.fit(bn_df, estimator=MaximumLikelihoodEstimator)
    learner = BayesianNetworkLearner()
    mb = learner._compute_markov_blankets(model)
    return EOSCModel(model=model, markov_blanket=mb)


# ─── Action map ────────────────────────────────────────────────────────────

STEEL_ACTIONS = {
    "tool_alignment":          {"pixels_area": "normal"},
    "cleaning_cycle":          {"edges_idx": "normal"},
    "adjust_rolling_pressure": {"orientation": "neutral"},
    "general_inspection":      {"quality_status": "good"},
}
for action, patch in STEEL_ACTIONS.items():
    vfe_mod.HEALING_ACTION_HYPOTHESIS[action] = patch

STEEL_ROOT_TO_ACTION = {
    "pixels_area":    "tool_alignment",
    "edges_idx":      "cleaning_cycle",
    "orientation":    "adjust_rolling_pressure",
    "log_area":       "tool_alignment",
    "quality_status": "general_inspection",
}
for root, action in STEEL_ROOT_TO_ACTION.items():
    tsg.ROOT_VARIABLE_TO_ACTION[root] = action

FAULT_TO_CORRECT_ACTION = {
    "K_Scatch": "tool_alignment",
    "Stains":   "cleaning_cycle",
    "Pastry":   "adjust_rolling_pressure",
}

ALL_ACTIONS = set(STEEL_ACTIONS.keys())


# ─── Agents ────────────────────────────────────────────────────────────────

class RuleBased:
    def process(self, obs):
        pa = float(obs.get("Pixels_Areas", 0))
        oi = float(obs.get("Orientation_Index", 0))
        ei = float(obs.get("Edges_Index", 0))
        # Priority: large area → tool; high edges → cleaning; positive orient → rolling
        if pa > 1000:
            return {"action": "tool_alignment"}
        if ei > 0.45:
            return {"action": "cleaning_cycle"}
        if oi > 0.3:
            return {"action": "adjust_rolling_pressure"}
        return {"action": "general_inspection"}


class AIFNoGate:
    def __init__(self, eosc_model):
        self.rca = RootCauseAnalyzer(eosc_model)

    def process(self, obs):
        obs_bn = discretize_observation(obs)
        obs_bn["slo_violated"] = True
        causes = self.rca.find_root_causes(
            symptom="slo_violated", symptom_value=True,
            observation=obs_bn, top_k=3,
        )
        if not causes:
            return {"action": None}
        top = causes[0]
        action = tsg.candidate_action_for_root_variable(top.variable)
        return {"action": action, "cause": top.variable,
                "cert": float(getattr(top, "posterior_prob", 0.0))}


class AURORA_Steel:
    def __init__(self, eosc_model, vfe_threshold=10.0, tau_base=0.70,
                 tau_min=0.50, tau_lambda=0.15):
        self.eosc_model = eosc_model
        self.rca = RootCauseAnalyzer(eosc_model)
        self.vfe_computer = VFEComputer()
        self.gate = UncertaintyGate(threshold=vfe_threshold)
        self.tau_base = tau_base
        self.tau_min = tau_min
        self.tau_lambda = tau_lambda
        self.vfe_threshold = vfe_threshold

    def process(self, obs):
        obs_bn = discretize_observation(obs)
        obs_bn["slo_violated"] = True
        n_anom = count_anomalies(obs)
        tau_eff = dynamic_tau_from_anomalies(n_anom, self.tau_base, self.tau_min, self.tau_lambda)
        diag = diagnose_slo_violation(self.rca, obs_bn, top_k=3, impact_calibration=0.42)
        if not diag["root_causes"]:
            return {"action": None, "abstained": True, "reason": "no_root_causes",
                    "gate1_fired": False, "gate2_fired": False}
        cert = diag["certainty"]
        action = diag["candidate_action"]
        cause = diag["top_cause"]
        if cert < tau_eff:
            return {"action": None, "abstained": True, "reason": "low_certainty",
                    "cert": cert, "cause": cause.variable,
                    "gate1_fired": True, "gate2_fired": False}
        vfe = self.vfe_computer.compute_vfe(self.eosc_model, obs_bn, action)
        if not self.gate.should_act(vfe):
            return {"action": None, "abstained": True, "reason": "high_vfe",
                    "vfe": vfe, "cert": cert,
                    "gate1_fired": False, "gate2_fired": True}
        return {"action": action, "abstained": False, "vfe": vfe, "cert": cert,
                "cause": cause.variable,
                "gate1_fired": False, "gate2_fired": False}


def classify_outcome(fault_type, result):
    if result.get("abstained"):
        return "abstained"
    action = result.get("action")
    if action is None:
        return "no_action"
    correct = FAULT_TO_CORRECT_ACTION[fault_type]
    if action == correct:
        return "correct"
    if action in ALL_ACTIONS:
        return "destructive"
    return "incorrect"


# ─── F_th auto-calibration ─────────────────────────────────────────────────

def calibrate_vfe_threshold(eosc_model, fault_trials):
    rca = RootCauseAnalyzer(eosc_model)
    vfe_computer = VFEComputer()
    safe_vfes, unsafe_vfes = [], []
    # Use ~40 trials per fault for calibration
    cal_rows = []
    for f in FAULT_CLASSES:
        cal_rows.extend(fault_trials[fault_trials["fault_label"] == f].head(40).to_dict(orient="records"))
    for row in cal_rows:
        obs = dict(row)
        obs["slo_violated"] = True
        obs_bn = discretize_observation(obs)
        diag = diagnose_slo_violation(rca, obs_bn, top_k=3, impact_calibration=0.42)
        if not diag["root_causes"]:
            continue
        action = diag["candidate_action"]
        vfe = vfe_computer.compute_vfe(eosc_model, obs_bn, action)
        correct = FAULT_TO_CORRECT_ACTION[row["fault_label"]]
        (safe_vfes if action == correct else unsafe_vfes).append(vfe)

    if not safe_vfes:
        return 10.0
    max_safe = max(safe_vfes)
    if unsafe_vfes and min(unsafe_vfes) > max_safe:
        F_th = max_safe + 0.3 * (min(unsafe_vfes) - max_safe)
        print(f"  F_th = {F_th:.3f} (GAP found: [{max_safe:.3f}, {min(unsafe_vfes):.3f}])")
        return F_th
    F_th = max_safe + 1.0
    print(f"  F_th = {F_th:.3f} (NO GAP)")
    print(f"    safe   n={len(safe_vfes)}  range [{min(safe_vfes):.3f}, {max(safe_vfes):.3f}]")
    if unsafe_vfes:
        print(f"    unsafe n={len(unsafe_vfes)} range [{min(unsafe_vfes):.3f}, {max(unsafe_vfes):.3f}]")
    return F_th


# ─── Experiment runner ─────────────────────────────────────────────────────

def run():
    out_dir = Path("/Users/suvidesilva/Downloads/ai_agnt/experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(lambda m: None)

    random.seed(42)
    np.random.seed(42)

    print("=" * 80)
    print("STEEL PLATES FAULTS — fourth workload sandbox")
    print("=" * 80)

    training_rows, fault_trials = load_dataset()
    print(f"  Training rows: {len(training_rows)}")
    print(f"  Test trials:   {len(fault_trials)} clean single-fault rows")
    for f in FAULT_CLASSES:
        n = len(fault_trials[fault_trials["fault_label"] == f])
        print(f"    {f:<12} n={n}")

    eosc_model = fit_steel_bn(training_rows)
    print(f"  BN: {len(STEEL_EDGES)} edges fitted")
    print()
    F_th = calibrate_vfe_threshold(eosc_model, fault_trials)
    print()

    agents = {
        "steel_rule_based":   RuleBased(),
        "steel_aif_no_gate":  AIFNoGate(eosc_model),
        "steel_aurora":       AURORA_Steel(eosc_model, vfe_threshold=F_th),
    }

    results = {}
    for name, agent in agents.items():
        per_fault = {f: Counter() for f in FAULT_CLASSES}
        all_trials = []
        gate1_fires = 0
        gate2_fires = 0
        for _, row in fault_trials.iterrows():
            fault = row["fault_label"]
            obs = dict(row)
            obs["slo_violated"] = True
            r = agent.process(obs)
            outcome = classify_outcome(fault, r)
            per_fault[fault][outcome] += 1
            if r.get("gate1_fired"): gate1_fires += 1
            if r.get("gate2_fired"): gate2_fires += 1
            all_trials.append({
                "fault_type": fault,
                "action_outcome": outcome,
                "action_taken": r.get("action"),
                "abstention": r.get("abstained", r.get("action") is None),
                "destructive_action": outcome == "destructive",
                "certainty_score": r.get("cert"),
                "vfe_value": r.get("vfe"),
                "root_cause_identified": r.get("cause"),
                "gate1_fired": r.get("gate1_fired", False),
                "gate2_fired": r.get("gate2_fired", False),
            })
        results[name] = {
            "per_fault": {f: dict(c) for f, c in per_fault.items()},
            "trials": all_trials,
            "gate1_fires": gate1_fires,
            "gate2_fires": gate2_fires,
        }
        out_path = out_dir / f"{name}_results.json"
        with open(out_path, "w") as f:
            json.dump({"trials": all_trials, "num_trials": len(all_trials)}, f)
        print(f"  Wrote {out_path.name}")

    n = len(fault_trials)
    print()
    print("=" * 80)
    print(f"STEEL PLATES FAULTS RESULTS  (F_th = {F_th:.2f}, n = {n})")
    print("=" * 80)
    print(f"{'Agent':<24} {'Correct':<12} {'Abstain':<12} {'Destructive':<12}")
    print("-" * 80)
    for name, data in results.items():
        agg = Counter()
        for c in data["per_fault"].values():
            agg.update(c)
        nn = sum(agg.values())
        print(f"{name:<24} {100*agg['correct']/nn:6.2f}%      "
              f"{100*agg['abstained']/nn:6.2f}%      "
              f"{100*agg['destructive']/nn:6.2f}%")

    print()
    print(f"AURORA gate firings:")
    print(f"  Gate 1: {results['steel_aurora']['gate1_fires']}")
    print(f"  Gate 2: {results['steel_aurora']['gate2_fires']}")

    print()
    print("Per-fault breakdown (AURORA):")
    for f, c in results["steel_aurora"]["per_fault"].items():
        nn = sum(c.values())
        if nn == 0: continue
        cells = "  ".join(f"{k}={100*v/nn:5.2f}%" for k, v in sorted(c.items()))
        print(f"  {f:<14} (n={nn}): {cells}")


if __name__ == "__main__":
    run()
