import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/agents/trajectory/tool_monitor.py
"""
Monitor and validate external tool/API calls.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Set

from sentinelml.core.base import BaseGuardrail


class ToolMonitor(BaseGuardrail):
    """
    Monitor safety and correctness of tool calls.

    Validates parameters, checks for dangerous operations,
    and monitors rate limits and errors.

    Parameters
    ----------
    allowed_tools : list
        Whitelist of allowed tools.
    dangerous_patterns : list
        Patterns indicating dangerous operations.
    rate_limits : dict
        Rate limits per tool.

    Examples
    --------
    >>> monitor = ToolMonitor(
    ...     allowed_tools=['search', 'calculator', 'weather'],
    ...     dangerous_patterns=['rm -rf', 'DROP TABLE']
    ... )
    >>> result = monitor.validate(tool_call)
    """

    def __init__(
        self,
        name: str = "ToolMonitor",
        fail_mode: str = "block",
        allowed_tools: Optional[List[str]] = None,
        dangerous_patterns: Optional[List[str]] = None,
        rate_limits: Optional[Dict[str, int]] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)
        self.allowed_tools = set(allowed_tools or [])
        self.dangerous_patterns = dangerous_patterns or [
            "rm -rf",
            "DROP TABLE",
            "DELETE FROM",
            "shutdown",
            "format",
            "exec(",
            "eval(",
            "os.system",
            "subprocess",
        ]
        self.rate_limits = rate_limits or {}
        self.tool_usage: Dict[str, List[float]] = {}

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def validate(self, tool_call: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate a tool call.

        Parameters
        ----------
        tool_call : dict
            With keys: 'tool', 'parameters', etc.
        context : dict, optional
            Additional context.

        Returns
        -------
        dict with validation results.
        """
        tool_name = tool_call.get("tool", "")
        parameters = tool_call.get("parameters", {})
        param_str = str(parameters)

        issues = []

        # Check if tool is allowed
        if self.allowed_tools and tool_name not in self.allowed_tools:
            issues.append(f"Tool '{tool_name}' not in allowed list")

        # Check for dangerous patterns in parameters
        for pattern in self.dangerous_patterns:
            if pattern.lower() in param_str.lower():
                issues.append(f"Dangerous pattern detected: {pattern}")

        # Check rate limits
        if tool_name in self.rate_limits:
            limit = self.rate_limits[tool_name]
            recent_calls = self._get_recent_calls(tool_name, window=60)  # 1 minute window
            if len(recent_calls) >= limit:
                issues.append(f"Rate limit exceeded for {tool_name}: {limit}/min")

        # Record usage
        self._record_usage(tool_name)

        is_valid = len(issues) == 0

        return {
            "is_valid": is_valid,
            "score": 1.0 if is_valid else 0.0,
            "action": "pass" if is_valid else self.fail_mode,
            "metadata": {
                "tool": tool_name,
                "issues": issues,
                "rate_limit_status": self._get_rate_limit_status(tool_name),
            },
        }

    def _get_recent_calls(self, tool_name: str, window: int) -> List[float]:
        """Get recent calls within time window."""
        if tool_name not in self.tool_usage:
            return []

        now = time.time()
        cutoff = now - window
        return [t for t in self.tool_usage[tool_name] if t > cutoff]

    def _record_usage(self, tool_name: str):
        """Record tool usage."""
        if tool_name not in self.tool_usage:
            self.tool_usage[tool_name] = []
        self.tool_usage[tool_name].append(time.time())

    def _get_rate_limit_status(self, tool_name: str) -> Dict[str, Any]:
        """Get current rate limit status."""
        if tool_name not in self.rate_limits:
            return {"limited": False}

        limit = self.rate_limits[tool_name]
        recent = self._get_recent_calls(tool_name, window=60)

        return {
            "limited": len(recent) >= limit,
            "used": len(recent),
            "limit": limit,
            "remaining": max(0, limit - len(recent)),
        }

    def reset_counters(self):
        """Reset all usage counters."""
        self.tool_usage = {}
