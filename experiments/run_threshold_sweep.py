"""
Threshold sweep for AURORA certainty/VFE gates.

Produces:
  - experiments/results/threshold_sweep_summary.json
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_comparison import ExperimentRunner


if __name__ == "__main__":
    runner = ExperimentRunner(num_trials=30, output_dir="experiments/results")
    rows = runner.run_threshold_sweep(
        certainty_thresholds=[0.70, 0.75, 0.80, 0.85, 0.90],
        vfe_thresholds=[1.0, 1.5, 2.0, 2.5, 3.0],
    )
    print(f"Sweep complete: {len(rows)} configs")
