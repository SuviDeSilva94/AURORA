"""
Edge ↔ fog handoff for computing-continuum demos (Sedlak-style).

The full thesis deployment would use real network calls (HTTP/gRPC/MQTT) from edge
to fog. Here we model that boundary explicitly so healing actions can be shown to
target a *fog tier* instead of only logging a generic stub.

Usage:
    from src.aif.continuum_bridge import FogTierReceiver, register_fog_offload_tools
    fog = FogTierReceiver(node_id="fog-001")
    register_fog_offload_tools(fog)
    # ... run MultiAgentCoordinator; offload_to_fog will call fog.receive_offload(...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from src.aif.healing_tools import ToolResult, get_healing_tool_registry


@dataclass
class FogTierReceiver:
    """
    In-process stand-in for a fog node API that accepts offload / processing handoffs.

    For production, replace method bodies with e.g. ``requests.post(fog_url, json=payload)``.
    """

    node_id: str = "fog-001"
    received: List[Dict[str, Any]] = field(default_factory=list)

    def receive_offload(self, params: Dict[str, Any]) -> ToolResult:
        """Record and acknowledge an edge → fog offload request."""
        payload = {
            "tier": "fog",
            "fog_node_id": self.node_id,
            "edge_context": params,
        }
        self.received.append(payload)
        logger.success(
            "[FOG {}] Accepted offload handoff from edge ({} keys in context)",
            self.node_id,
            len(params),
        )
        return ToolResult(
            action="offload_to_fog",
            success=True,
            message=f"handed_off_to_{self.node_id}",
            details=payload,
        )


def register_fog_offload_tools(receiver: FogTierReceiver) -> None:
    """Wire ``offload_to_fog`` and ``offload_processing`` to the fog receiver."""
    reg = get_healing_tool_registry()

    def _offload(params: Dict[str, Any]) -> ToolResult:
        return receiver.receive_offload(params)

    reg.register("offload_to_fog", _offload)
    reg.register("offload_processing", _offload)
    logger.info("[Continuum] Healing tools offload_to_fog → FogTierReceiver({})", receiver.node_id)


def make_http_offload_tool(url: str, headers: Optional[Dict[str, str]] = None) -> Callable[[Dict[str, Any]], ToolResult]:
    """
    Factory for a real distributed offload (optional).

    Example::

        import requests
        reg = get_healing_tool_registry()
        reg.register("offload_to_fog", make_http_offload_tool("https://fog.local/v1/offload"))

    Requires ``requests`` in the environment.
    """
    import requests

    hdrs = headers or {}

    def _call(params: Dict[str, Any]) -> ToolResult:
        try:
            r = requests.post(url, json=params, headers=hdrs, timeout=10)
            return ToolResult(
                action="offload_to_fog",
                success=r.ok,
                message=r.text[:500],
                details={"status_code": r.status_code, "params": params},
            )
        except Exception as e:
            logger.exception("HTTP offload failed: {}", e)
            return ToolResult(
                action="offload_to_fog",
                success=False,
                message=str(e),
                details=params,
            )

    return _call
