# Experimental framework

Compares the proposed **AURORA** pipeline (certainty on root cause + VFE before execution) with **rule-based** and **AIF-no-gate** baselines.

## Layout

```
experiments/
├── run_comparison.py    # Main runner
├── analyze_results.py   # Tables, charts, thesis_summary.txt
└── results/             # Created on run
    ├── rule_based_results.json
    ├── aif_no_gate_results.json
    ├── aurora_results.json
    ├── comparison_summary.json
    ├── comparison_chart.png
    ├── fault_type_breakdown.png
    ├── comparison_table.csv
    └── thesis_summary.txt
```

## Agents

| Agent | Implementation | Role |
|--------|----------------|------|
| Rule-based | `src/baselines/rule_based_agent.py` | Threshold rules, no BN |
| AIF-no-gate | `src/baselines/aif_no_gate_agent.py` | BN + RCA; **no** VFE safe-execution gate (acts when it proposes an action); no 85% certainty gate like AURORA |
| AURORA | `experiments/run_comparison.py` (wrapper) | Certainty threshold + VFE gate before acting |

## Fault types (default runner)

The default `ExperimentRunner.fault_types` uses three injectors:

1. **network_drop** — degraded network / bandwidth-style stress  
2. **cpu_spike** — high CPU  
3. **memory_leak** — high memory  

`FaultInjector` also defines **delay_spike**, **packet_loss**, and **overloaded_streams** helpers in `run_comparison.py`; they are **not** wired into the default comparison loop unless you extend `fault_types`.

## Metrics

Collected in `src/evaluation/metrics.py` (e.g. repair accuracy, MTTR, abstention, destructive actions, resolution rate where applicable).

## Design (matches `run_comparison.py`)

- Default **`num_trials=50`** per fault type **per agent**.  
- **3** fault types × **50** trials × **3** agents ⇒ **450** total trials (longer than a quick smoke test).  
- Training data for AIF agents: default **100** synthetic samples per run setup in code.

## Usage

From **project root** (venv activated, dependencies installed):

```bash
python3 experiments/run_comparison.py
python3 experiments/analyze_results.py
```

Use `python3` on macOS if `python` is not available.

## Statistical tests

`analyze_results.py` can run **t-tests** when per-trial detail is available; if only aggregated JSON exists, significance steps may be skipped (see logs).

## Thesis artifacts

After `analyze_results.py`, use:

- `experiments/results/comparison_table.csv`  
- `experiments/results/comparison_chart.png`  
- `experiments/results/fault_type_breakdown.png`  
- `experiments/results/thesis_summary.txt`  

**Note:** Numeric values in `thesis_summary.txt` depend on the last run (seed, `num_trials`, code version). Regenerate after changing the runner.

## Smoke test

```bash
python3 -m pytest tests/test_experimental_framework.py -v
```

---

See **`README.md`** (root), **`HOW_IT_WORKS.md`**, and **`CODE_EXPLANATION.md`** for architecture and paper mapping.
