"""
Main entry point for Distributed Agentic Micro-Agent for Resilience in the Computing Continuum

Thesis: Stockholm University,   Praveen Kumar Donta.
Runs the CCTV-edge pipeline: congestion detection → root cause (high certainty) → mitigation.
"""

import sys
from pathlib import Path

# Ensure project root on path when run as python src/main.py
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    """Run the thesis demo (congestion → root cause with high certainty → mitigation)."""
    from run_demo import main as run_demo_main
    run_demo_main()


if __name__ == "__main__":
    main()
