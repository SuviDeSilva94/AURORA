# Distributed Agentic Micro-Agent for Resilience in the Computing Continuum

Thesis project: **Distributed Agentic Micro-Agent for Resilience in the Computing Continuum**  
** :** Praveen Kumar Donta  
**Foundation papers:** Sedlak et al. (2024), Donta et al. (2025)

This codebase implements:

- **pgmpy** Bayesian Networks for causal inference (Sedlak-style EOSC model)
- **Separate certainty check** on root cause (≥85% required; no uncertainty on final decision)
- **LangGraph** stateful multi-agent orchestration with parallel micro-agents
- **Fault injection:** synthetic network congestion (delay, packet loss, overloaded streams)
- **Metrics:** repair accuracy, MTTR, resolution rate
- **Experiments:** 50–100 trials, comparison table, charts, `thesis_summary.txt`

---

## Setup

### 1. Clone and enter project

```bash
cd /path/to/ai_agnt
```

### 2. Create virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install project in editable mode (for `src` imports)

```bash
pip install -e .
```

If you skip editable install, run from the project root (demo scripts add the root to `sys.path`).

---

## Documentation (Markdown)

| File | Contents |
|------|----------|
| `README.md` | Setup, run commands, project layout |
| `HOW_IT_WORKS.md` | End-to-end pipeline and file map |
| `CODE_EXPLANATION.md` | Sedlak / Donta mapping, components, certainty check locations |
| `SCOPE_AND_NEXT_STEPS.md` | Simulated vs real deployment; Pi / fog / HTTP next steps |
| `THESIS_IMPLEMENTATION_GUIDE.md` | **Thesis ↔ code alignment**, component catalog, copy-paste text for research document |
| `experiments/README.md` | Experiment runner, agents, metrics, outputs |

---

## Run Commands

**Install dependencies first** (Setup above). If you see `ModuleNotFoundError: No module named 'loguru'`, you are using a Python that does not have the project packages — run `pip install -r requirements.txt` inside your venv (or `python3 -m pip install -r requirements.txt` if you skip the venv, not recommended).

**If you see `command not found: python` (common on macOS):** use `python3` instead, or activate the venv first (`source venv/bin/activate`) — then `python` works.

### Working demo: congestion → root cause → mitigation

From project root:

```bash
python3 demo_01_basic_pipeline_test.py
```

This runs: congestion detection → certain root-cause identification (Bayesian Network + certainty check) → VFE-gated mitigation (or abstention).

### Real-world scenario: Raspberry Pi CCTV + network congestion

Uses the exact thesis setup (30 FPS, &lt;33 ms, &gt;1.6 MB/s; congestion: 18 FPS, 80 ms, 0.9 MB/s):

```bash
python3 demo_02_cctv_edge_scenario.py
```

- **Default:** Root cause is identified (e.g. `bandwidth_low`); if certainty &lt; 85% the system abstains (no mitigation).
- **Optional:** `DEMO_MITIGATION=1 python3 demo_02_cctv_edge_scenario.py` uses a lower certainty threshold so you see the full flow including mitigation (e.g. offload_to_fog).

### Computing continuum (edge → fog handoff + FogNode)

Structured demo: edge `MultiAgentCoordinator`, healing tools calling a **fog-tier receiver**, then a short **FogNode** cluster loop (Sedlak-style):

```bash
python3 demo_03_continuum_fog_handoff.py
```

Use `STRICT=1 python3 demo_03_continuum_fog_handoff.py` for 85% certainty (may abstain). See **`SCOPE_AND_NEXT_STEPS.md`** for real HTTP/Pi integration.

### CCTV scenario demo

```bash
python3 tests/demo_cctv_scenario.py
```

### Other demos

```bash
python3 tests/demo_multi_agent.py
python3 tests/demo_certainty.py
```

### Experiments (50–100 trials, comparison table + charts + thesis summary)

Run comparison over rule-based, AIF-no-gate, and AURORA agents:

```bash
python3 experiments/run_comparison.py
```

Then generate comparison table, charts, and thesis summary:

```bash
python3 experiments/analyze_results.py
```

Outputs:

- `experiments/results/comparison_table.csv`
- `experiments/results/comparison_chart.png`
- `experiments/results/fault_type_breakdown.png`
- `experiments/results/thesis_summary.txt`

### Tests

```bash
python3 -m pytest tests/ -v
```

---

## Project layout

| Path | Purpose |
|------|--------|
| `src/aif/bayesian_network.py` | pgmpy BN, EOSC model, Markov blankets (Sedlak 2024) |
| `src/aif/root_cause_analyzer.py` | Causal diagnosis, root cause discovery |
| `src/aif/vfe_compute.py` | VFE and safe-execution threshold (Donta 2025) |
| `src/aif/certainty_check_agent.py` | Dedicated agent: “is the root-cause decision certain?” (certainty + VFE) |
| `src/aif/healing_tools.py` | Functional tool calling for self-healing (register real APIs for production) |
| `src/aif/multi_agent_system.py` | Multi-agent pipeline + **certainty check** on root cause |
| `src/aif/multi_agent_certainty.py` | Certainty aggregation and thresholds |
| `src/aif/langgraph_orchestrator.py` | LangGraph stateful orchestration |
| `src/aif/specialized_agents.py` | CCTV micro-agents (bandwidth, camera, node, delay) |
| `src/evaluation/metrics.py` | Repair accuracy, MTTR, resolution rate |
| `experiments/run_comparison.py` | Fault injection and experiment runner |
| `HOW_IT_WORKS.md` | Plain-language pipeline and component reference |
| `CODE_EXPLANATION.md` | Paper mapping, components, certainty / VFE locations |
| `SCOPE_AND_NEXT_STEPS.md` | Simulated vs deployed continuum; Pi/fog extensions |
| `THESIS_IMPLEMENTATION_GUIDE.md` | Title/subtitle ↔ code; thesis sections; paste-ready paragraphs |
| `src/aif/continuum_bridge.py` | Fog handoff + optional HTTP tool for real fog API |
| `src/utils/pi_metrics_hook.py` | Template: real FPS/delay/throughput → observation dict |

---

## Technical rules (summary)

- **pgmpy** for Bayesian Networks and causal inference (Sedlak prototype).
- **Certainty check:** high certainty required on root cause; no acting under uncertainty on the final decision (see `CODE_EXPLANATION.md`).
- **LangGraph** for stateful multi-agent orchestration; parallel micro-agents in RCA phase.
- **Fault injection:** synthetic network congestion (delay, packet loss, overloaded streams) in `experiments/run_comparison.py`.
- **Metrics:** repair accuracy, MTTR, resolution rate (and optionally abstention/destructive).
- **Experiments:** 50–100 trials (default 50 per fault type), comparison table, charts, `thesis_summary.txt`.

All code is intended to be clean, commented, and suitable for thesis use.
