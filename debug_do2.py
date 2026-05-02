import numpy as np
from src.aif.bayesian_network import fit_eosc_cctv_benchmark_dag
from pgmpy.inference import VariableElimination, CausalInference
import networkx as nx

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
model = fit_eosc_cctv_benchmark_dag(rows)

ci = CausalInference(model.model)
adj_set = ci.get_all_backdoor_adjustment_sets("network_quality", "slo_violated")
print("Adj set for nq -> slo:", adj_set)

# What is pgmpy doing inside query?
# If we just do graph mutilation ourselves:
mutilated = model.model.do(["network_quality", "delay"])
ve_mut = VariableElimination(mutilated)
obs = {"memory": "normal", "cpu": "high"}
do_set = {"network_quality": "high", "delay": "normal"}
res_mut = ve_mut.query(["slo_violated"], evidence={**obs, **do_set})
print("Manual mutilation:")
print(res_mut)
