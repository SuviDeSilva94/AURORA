"""
Optional hook: replace synthetic CCTV metrics with real Raspberry Pi / camera readings.

The thesis codebase uses dict observations (FPS, delay, throughput, flags). On a real Pi,
fill the same keys from:

- FPS: camera pipeline / OpenCV capture loop timing
- delay: RTT to ingest server or frame timestamp skew
- throughput: ``n_bytes / interval`` on the uplink interface
- ``*_bad`` / ``slo_violated``: compare to thresholds in ``src.utils.cctv_config``

Example (pseudo-code)::

    from src.utils.cctv_config import CCTV_SLOS
    from src.utils.pi_metrics_hook import build_observation_from_pi

    obs = build_observation_from_pi(
        fps=measure_fps(),
        delay_ms=measure_rtt_ms(),
        throughput_mbps=measure_uplink_mbps(),
    )
    coordinator.process_observation(obs)

This module keeps **stubs** so the repo runs without hardware; copy the pattern into your Pi script.
"""

from __future__ import annotations

from typing import Any, Dict


def measure_fps_stub() -> float:
    """Replace with real FPS from your camera pipeline."""
    return 30.0


def measure_delay_ms_stub() -> float:
    """Replace with ping or application-level latency."""
    return 20.0


def measure_throughput_mbps_stub() -> float:
    """Replace with interface stats (e.g. read /proc/net/dev deltas)."""
    return 2.0


def build_observation_from_pi(
    fps: float,
    delay_ms: float,
    throughput_mbps: float,
    cpu: float = 50.0,
    memory: float = 55.0,
    camera_id: str = "camera_1",
    node_id: str = "raspberry-pi-001",
    slo_fps: float = 30.0,
    slo_delay_max: float = 33.0,
    slo_throughput_min: float = 1.6,
) -> Dict[str, Any]:
    """
    Build an observation dict compatible with ``MultiAgentCoordinator`` / CCTV demos.

    Thresholds default to thesis SLOs; adjust to match ``CCTV_SLOS`` in ``cctv_config.py``.
    """
    fps_bad = 1 if fps < slo_fps * 0.95 else 0
    delay_bad = 1 if delay_ms > slo_delay_max else 0
    throughput_bad = 1 if throughput_mbps < slo_throughput_min else 0
    slo_violated = 1 if (fps_bad or delay_bad or throughput_bad) else 0
    return {
        "fps": fps,
        "delay": delay_ms,
        "throughput": throughput_mbps,
        "cpu": cpu,
        "memory": memory,
        "camera_id": camera_id,
        "node_id": node_id,
        "fps_bad": fps_bad,
        "delay_bad": delay_bad,
        "throughput_bad": throughput_bad,
        "bandwidth_low": throughput_bad,
        "node_overload": 0,
        "slo_violated": slo_violated,
    }
