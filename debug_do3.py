import numpy as np
from src.aif.bayesian_network import fit_eosc_cctv_benchmark_dag
from pgmpy.inference import VariableElimination, CausalInference

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
obs = {"memory": "normal", "cpu": "high"}
do_set = {"network_quality": "high", "delay": "normal"}
res_do = ci.query(["slo_violated"], do=do_set, evidence=obs, adjustment_set=[])
print(res_do)
