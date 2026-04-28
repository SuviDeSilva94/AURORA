"""
Functional tool calling for self-healing.

Registry of callables for each healing action. Executor invokes these; default
implementations are stubs that log. In production, register real API calls, e.g.:

    from src.aif.healing_tools import get_healing_tool_registry, ToolResult
    def my_offload(params):
        r = requests.post("https://fog-api/offload", json=params)
        return ToolResult("offload_to_fog", r.ok, r.text, {"status_code": r.status_code})
    get_healing_tool_registry().register("offload_to_fog", my_offload)
"""

from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ToolResult:
    """Result of invoking one healing tool."""
    action: str
    success: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


# Default (stub) implementations — log and return success for simulation
def _stub_offload_to_fog(params: Dict[str, Any]) -> ToolResult:
    logger.info("[TOOL] offload_to_fog: would call fog offload API with params {}", params)
    return ToolResult(action="offload_to_fog", success=True, message="stub executed", details=params)


def _stub_reduce_bitrate(params: Dict[str, Any]) -> ToolResult:
    logger.info("[TOOL] reduce_bitrate: would set encoder bitrate with params {}", params)
    return ToolResult(action="reduce_bitrate", success=True, message="stub executed", details=params)


def _stub_rate_limit_camera(params: Dict[str, Any]) -> ToolResult:
    logger.info("[TOOL] rate_limit_camera: would rate-limit camera with params {}", params)
    return ToolResult(action="rate_limit_camera", success=True, message="stub executed", details=params)


def _stub_reduce_fps(params: Dict[str, Any]) -> ToolResult:
    logger.info("[TOOL] reduce_fps: would reduce FPS with params {}", params)
    return ToolResult(action="reduce_fps", success=True, message="stub executed", details=params)


def _stub_redistribute_load(params: Dict[str, Any]) -> ToolResult:
    logger.info("[TOOL] redistribute_load: would redistribute load with params {}", params)
    return ToolResult(action="redistribute_load", success=True, message="stub executed", details=params)


def _stub_maintain(params: Dict[str, Any]) -> ToolResult:
    logger.info("[TOOL] maintain: no-op (monitor only)")
    return ToolResult(action="maintain", success=True, message="no-op", details=params)


def _stub_generic(action: str, params: Dict[str, Any]) -> ToolResult:
    logger.info("[TOOL] {}: generic stub with params {}", action, params)
    return ToolResult(action=action, success=True, message="stub executed", details=params)


def _make_generic_stub(action: str) -> Callable[[Dict[str, Any]], ToolResult]:
    return lambda params: _stub_generic(action, params)


class HealingToolRegistry:
    """
    Registry of callables for each healing action (functional tool calling).
    Register real implementations (e.g. HTTP to fog, K8s API) for production.
    """

    def __init__(self):
        self._tools: Dict[str, Callable[[Dict[str, Any]], ToolResult]] = {
            "offload_to_fog": _stub_offload_to_fog,
            "reduce_bitrate": _stub_reduce_bitrate,
            "throttle_connections": _make_generic_stub("throttle_connections"),
            "switch_network": _make_generic_stub("switch_network"),
            "rate_limit_camera": _stub_rate_limit_camera,
            "reduce_fps": _stub_reduce_fps,
            "reduce_resolution": _make_generic_stub("reduce_resolution"),
            "offload_processing": _stub_offload_to_fog,
            "clear_cache": _make_generic_stub("clear_cache"),
            "restart_service": _make_generic_stub("restart_service"),
            "reduce_buffer_size": _make_generic_stub("reduce_buffer_size"),
            "redistribute_load": _stub_redistribute_load,
            "maintain": _stub_maintain,
            "monitor": _stub_maintain,
        }

    def register(self, action: str, fn: Callable[[Dict[str, Any]], ToolResult]) -> None:
        """Register a real implementation for an action (e.g. API call)."""
        self._tools[action] = fn
        logger.info(f"[HealingTools] Registered tool for action: {action}")

    def execute(
        self,
        action: str,
        root_cause: Optional[str] = None,
        observation: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Invoke the tool for the given action (functional tool calling).
        """
        params = {"root_cause": root_cause, "observation": observation or {}, **kwargs}
        fn = self._tools.get(action)
        if fn is None:
            return _stub_generic(action, params)
        try:
            return fn(params)
        except Exception as e:
            logger.exception(f"[HealingTools] Tool {action} failed: {e}")
            return ToolResult(
                action=action,
                success=False,
                message=str(e),
                details=params,
            )


_default_registry: Optional[HealingToolRegistry] = None


def get_healing_tool_registry() -> HealingToolRegistry:
    """Return the default healing tool registry (lazy init)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = HealingToolRegistry()
    return _default_registry


def reset_healing_tool_registry() -> None:
    """Clear the singleton registry (e.g. before a demo that re-registers fog tools)."""
    global _default_registry
    _default_registry = None
