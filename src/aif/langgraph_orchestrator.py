"""
LangGraph Stateful Multi-Agent Orchestration

Uses LangGraph StateGraph for the self-healing pipeline:
  detect -> RCA (parallel) -> certainty_check -> plan -> vfe_gate -> execute

State is carried across nodes; parallel micro-agents run in the RCA phase.
References: Sedlak et al. (2024) causal diagnosis, Donta et al. (2025) VFE gating.
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from loguru import logger

# LangGraph imports
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    START = END = None

from src.aif.bayesian_network import EOSCModel
from src.aif.root_cause_analyzer import RootCauseAnalyzer, RootCause
from src.aif.vfe_compute import VFEComputer, UncertaintyGate


# ─── State schema for the graph ───────────────────────────────────────────

class SelfHealingState(TypedDict, total=False):
    """State passed between LangGraph nodes. All fields optional for incremental updates."""
    observation: Dict[str, Any]
    anomalies: List[Any]
    root_causes: List[Any]
    top_root_cause: Optional[str]
    certainty: float
    passed_certainty: bool
    solutions: List[Any]
    executed: List[Dict[str, Any]]
    action_taken: Optional[str]
    abstained: bool
    reason: Optional[str]
    vfe: Optional[float]
    parallel_findings: List[Any]


# ─── Node implementations (each can be wired to coordinator agents) ─────────

def _node_detect(state: SelfHealingState, detector) -> SelfHealingState:
    """Node: run detector agent on observation."""
    obs = state.get("observation") or {}
    anomalies = detector.detect_anomalies(obs)
    return {"anomalies": anomalies}


def _node_rca(state: SelfHealingState, rca_agents: List[Any]) -> SelfHealingState:
    """Node: run RCA agents (parallel in practice) and aggregate root causes."""
    obs = state.get("observation") or {}
    anomalies = state.get("anomalies") or []
    if not anomalies:
        return {"root_causes": []}
    all_causes: List[RootCause] = []
    for rca in rca_agents:
        causes = rca.find_root_causes(anomalies, obs)
        all_causes.extend(causes)
    # Aggregate: dedupe by variable, boost if multiple agree
    agg: Dict[str, RootCause] = {}
    for rc in all_causes:
        if rc.variable not in agg:
            agg[rc.variable] = rc
        else:
            ex = agg[rc.variable]
            ex.impact_score = (ex.impact_score + rc.impact_score) / 2 * 1.2
            ex.evidence_strength = max(ex.evidence_strength, rc.evidence_strength)
    root_causes = list(agg.values())
    root_causes.sort(key=lambda r: r.impact_score, reverse=True)
    return {"root_causes": root_causes}


def _node_certainty_check(
    state: SelfHealingState,
    certainty_threshold: float,
) -> SelfHealingState:
    """Node: separate certainty check on root cause (no uncertainty on final decision)."""
    root_causes = state.get("root_causes") or []
    if not root_causes:
        return {
            "top_root_cause": None,
            "certainty": 0.0,
            "passed_certainty": False,
            "abstained": True,
            "reason": "no_root_cause_found",
        }
    top = root_causes[0]
    certainty = (top.impact_score + top.evidence_strength) / 2.0
    passed = certainty >= certainty_threshold
    return {
        "top_root_cause": top.variable,
        "certainty": certainty,
        "passed_certainty": passed,
        "abstained": not passed,
        "reason": None if passed else "certainty_below_threshold",
    }


def _node_plan(state: SelfHealingState, planner) -> SelfHealingState:
    """Node: propose solutions for the top root cause."""
    if state.get("abstained"):
        return {"solutions": []}
    root_causes = state.get("root_causes") or []
    obs = state.get("observation") or {}
    solutions = planner.propose_solutions(root_causes, obs)
    return {"solutions": solutions}


def _node_vfe_gate(
    state: SelfHealingState,
    vfe_computer: VFEComputer,
    gate: UncertaintyGate,
    eosc_model: EOSCModel,
) -> SelfHealingState:
    """Node: VFE check (Donta 2025). Only allow execution if VFE < threshold."""
    if state.get("abstained") or not state.get("solutions"):
        return {"vfe": None}
    obs = state.get("observation") or {}
    try:
        vfe = vfe_computer.compute_vfe(eosc_model, obs, action={})
    except Exception:
        vfe = 2.5
    should_act = gate.should_act(vfe)
    if not should_act:
        return {
            "vfe": vfe,
            "abstained": True,
            "reason": "vfe_above_threshold",
            "executed": [],
            "action_taken": None,
        }
    return {"vfe": vfe}


def _node_execute(state: SelfHealingState, executor) -> SelfHealingState:
    """Node: execute solutions (executor already applies VFE internally; we gate above)."""
    if state.get("abstained"):
        return {"executed": [], "action_taken": None}
    solutions = state.get("solutions") or []
    obs = state.get("observation") or {}
    executed = executor.execute_solutions(solutions, obs)
    action_taken = solutions[0].action if solutions else None
    return {"executed": executed, "action_taken": action_taken}


# ─── Build the graph (when LangGraph is available) ──────────────────────────

def build_self_healing_graph(
    detector,
    rca_agents: List[Any],
    planner,
    executor,
    eosc_model: EOSCModel,
    certainty_threshold: float = 0.85,
    vfe_threshold: float = 2.0,
):
    """
    Build LangGraph StateGraph for: detect -> RCA -> certainty_check -> plan -> vfe_gate -> execute.

    Returns:
        Compiled graph (invoke with state dict), or None if LangGraph not available.
    """
    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not installed; use MultiAgentCoordinator instead.")
        return None

    vfe_computer = VFEComputer()
    gate = UncertaintyGate(threshold=vfe_threshold)

    builder = StateGraph(SelfHealingState)

    # Nodes
    builder.add_node(
        "detect",
        lambda s: _node_detect(s, detector),
    )
    builder.add_node(
        "rca",
        lambda s: _node_rca(s, rca_agents),
    )
    builder.add_node(
        "certainty_check",
        lambda s: _node_certainty_check(s, certainty_threshold),
    )
    builder.add_node(
        "plan",
        lambda s: _node_plan(s, planner),
    )
    builder.add_node(
        "vfe_gate",
        lambda s: _node_vfe_gate(s, vfe_computer, gate, eosc_model),
    )
    builder.add_node(
        "execute",
        lambda s: _node_execute(s, executor),
    )

    # Edges
    builder.add_edge(START, "detect")
    builder.add_edge("detect", "rca")
    builder.add_edge("rca", "certainty_check")
    builder.add_edge("certainty_check", "plan")
    builder.add_edge("plan", "vfe_gate")
    builder.add_edge("vfe_gate", "execute")
    builder.add_edge("execute", END)

    return builder.compile()


def run_langgraph_pipeline(
    graph,
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the compiled LangGraph pipeline for one observation.

    Returns a result dict compatible with MultiAgentCoordinator.process_observation.
    """
    if graph is None:
        return {
            "phase": "error",
            "status": "langgraph_unavailable",
            "abstained": True,
            "reason": "langgraph_unavailable",
        }
    initial: SelfHealingState = {"observation": observation}
    final = graph.invoke(initial)
    # Merge state into a single result
    return {
        "phase": "complete",
        "status": "processed",
        "observation": observation,
        "anomalies": final.get("anomalies", []),
        "root_causes": final.get("root_causes", []),
        "root_cause": final.get("top_root_cause"),
        "certainty": final.get("certainty", 0.0),
        "passed_certainty": final.get("passed_certainty", False),
        "solutions": final.get("solutions", []),
        "executed": final.get("executed", []),
        "action_taken": final.get("action_taken"),
        "abstained": final.get("abstained", True),
        "reason": final.get("reason"),
        "vfe": final.get("vfe"),
        "parallel_findings": final.get("parallel_findings", []),
    }
