# AURORA: An Uncertainty-Aware Resilience Micro-agent for Causal Observability in the Computing Continuum

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Thesis](https://img.shields.io/badge/Academic-Thesis-success.svg)](#citation--use-this-research)

**Official Codebase for the Research Paper:** *"AURORA: An Uncertainty-Aware Resilience Micro-agent for Causal Observability in the Computing Continuum"*

## 📖 Overview

The **Computing Continuum** integrates resource-constrained edge devices (like Raspberry Pi CCTV cameras), fog nodes, and cloud resources. However, when **grey failures** (ambiguous degradations like network congestion or memory leaks) occur, traditional rule-based resilience mechanisms often respond by blindly executing container restarts or threshold adjustments. This blind execution frequently exacerbates the fault, leading to catastrophic cascading failures.

**AURORA** is a framework of lightweight, parallel micro-agents that diagnose and resolve these anomalies through **causal inference and bounded uncertainty**, running natively on resource-constrained edge hardware. 

The core innovation of AURORA is the **Dual-Gated Safety Mechanism** and the **Abstention Paradigm**. The system mathematically guarantees that healing actions only execute when:
1. The inferred root cause exceeds a rigorous **posterior probability threshold ($\ge 0.85$)**.
2. The projected epistemic surprise remains below a maximum **Variational Free Energy (VFE) limit ($< 2.0$)**.

If either gate fails, the agent formally chooses inaction (intentional abstention) over blind execution, safely offloading the ambiguous payload to a fog-tier node.

---

## 🔬 Key Achievements & Experimental Results

The framework was evaluated through a rigorous **1,350-trial Monte Carlo simulation** across three controlled grey fault scenarios (network drop, CPU spike, memory leak). AURORA's performance was evaluated against a standard Rule-Based heuristic agent and an Un-gated Active Inference agent.

**The results fundamentally establish the "Abstention Paradigm" for safe continuum resilience:**

| Agent Type | Repair Accuracy | Resolution Rate | Destructive Actions | Abstention Rate | MTTR |
|---|---|---|---|---|---|
| **Rule-Based** | 25.9% | 45.3% | **33.3%** | 0.0% | 0.00s |
| **AIF (No Gate)** | 66.7% | 66.7% | **33.3%** | 0.0% | 0.01s |
| **AURORA (Proposed)**| 63.0% | 36.7% | **0.0%** | **63.3%** | 0.01s |

* **0.0% Destructive Action Rate:** While baselines routinely guessed during high uncertainty (causing 33.3% destructive actions that actively harmed the system), AURORA perfectly trapped uncertainty and completely eliminated harmful autonomous actions.
* **Extreme Efficiency:** Despite calculating Bayesian posteriors, Markov Blankets, and VFE limits, AURORA maintained a Mean Time to Repair (MTTR) of **~0.01 seconds**, proving the computational viability of Active Inference on edge-tier hardware.

---

## 🛠 Technologies Used

- **Active Inference (AIF):** Grounded in Friston's Free-Energy Principle, used as the cognitive model for causal diagnosis and mitigating epistemic surprise.
- **Bayesian Networks & do-Calculus:** Utilizing `pgmpy` and Bayesian Network Structure Learning (BNSL) to map localized causal state-graphs.
- **Markov Blankets:** To dramatically reduce computational overhead by limiting inference solely to a node's immediate causal neighborhood.
- **Multi-Agent Orchestration:** Utilizing `LangGraph` for stateful, parallelized micro-agent execution across different edge-system facets.

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/SuviDeSilva94/AURORA-activeinference-causal-observable-microagent-for-continuum-resilience.git
cd AURORA-activeinference-causal-observable-microagent-for-continuum-resilience
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install project in editable mode
```bash
pip install -e .
```

---

## 💻 Running the Demos

The repository contains several pre-configured demonstrations mapping to specific edge-computing scenarios.

### Working Demo: Congestion → Root Cause → Mitigation
Runs the basic pipeline: congestion detection → causal identification (Bayesian Network) → VFE-gated mitigation evaluation.
```bash
python3 demo_01_basic_pipeline_test.py
```

### Real-world Scenario: Raspberry Pi CCTV + Network Congestion
Simulates a live CCTV stream (30 FPS, <33 ms delay, >1.6 MB/s). Introduces a grey failure (18 FPS, 80 ms delay, 0.9 MB/s) to test the agent's diagnostic pipeline.
```bash
python3 demo_02_cctv_edge_scenario.py
```
*(Tip: Use `DEMO_MITIGATION=1 python3 demo_02_cctv_edge_scenario.py` to lower the certainty threshold and force a physical mitigation execution).*

### Computing Continuum: Edge → Fog Handoff
Demonstrates the Abstention Paradigm. When edge uncertainty is too high, the system packages the telemetry and formally offloads the diagnostic task to a simulated Fog Node.
```bash
python3 demo_03_continuum_fog_handoff.py
```

---

## 📊 Running the Monte Carlo Experiments

To reproduce the exact findings, tables, and graphs featured in the research paper (1,350 total trials), utilize the experimental framework:

**1. Run the fault-injection simulation:**
```bash
python3 experiments/run_comparison.py
```

**2. Generate the thesis data (Tables, CSVs, PNG Charts):**
```bash
python3 experiments/analyze_results.py
```
*Outputs will be saved directly to the `experiments/results/` directory.*

---

## 📚 Citation & Use This Research

If you use AURORA, its Dual-Gated framework, or the underlying Active Inference edge-implementations in your research, please cite this repository and the associated paper.

We actively encourage researchers in the fields of **IoT Resilience, Edge/Fog Computing, and Active Inference** to fork this repository, integrate real-world HTTP Fog-Node APIs via the `continuum_bridge.py`, and test the micro-agents on physical Raspberry Pi clusters.

```bibtex
@article{desilva2026aurora,
  title={AURORA: An Uncertainty-aware Resilience Micro-agent for Causal Observability in the Computing Continuum},
  author={De Silva, Suvi and Lapkovskis, Alfreds and Donta, Praveen Kumar},
  journal={Department of Computer Systems and Sciences, Stockholm University},
  year={2026}
}
```

**Contact:** thde1580@student.su.se | {alfreds.lapkovskis, praveen}@dsv.su.se
