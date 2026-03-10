import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/agents/state/budget_manager.py
"""
Token and cost budget management for agents.
"""

import time
from typing import Any, Callable, Dict, Optional

from sentinelml.core.base import BaseGuardrail


class BudgetManager(BaseGuardrail):
    """
    Monitor and enforce token/cost budgets.

    Prevents runaway costs by tracking usage and
    enforcing limits on tokens, API calls, and time.

    Parameters
    ----------
    token_budget : int, optional
        Maximum tokens allowed.
    cost_budget : float, optional
        Maximum cost in dollars.
    time_budget_seconds : int, optional
        Maximum execution time.

    Examples
    --------
    >>> budget = BudgetManager(token_budget=4000, cost_budget=0.50)
    >>> for step in trajectory:
    ...     result = budget.validate(step, context={'tokens_used': step_tokens})
    ...     if not result['is_valid']:
    ...         break
    """

    def __init__(
        self,
        name: str = "BudgetManager",
        fail_mode: str = "block",
        token_budget: Optional[int] = None,
        cost_budget: Optional[float] = None,
        time_budget_seconds: Optional[int] = None,
        token_cost_per_1k: float = 0.002,
        verbose: bool = False,
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)
        self.token_budget = token_budget
        self.cost_budget = cost_budget
        self.time_budget_seconds = time_budget_seconds
        self.token_cost_per_1k = token_cost_per_1k

        self.tokens_used = 0
        self.cost_incurred = 0.0
        self.start_time = time.time()

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def validate(self, step: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Check budget constraints.

        Parameters
        ----------
        step : dict
            Current step information.
        context : dict, optional
            Should contain 'tokens_used', 'cost', etc.

        Returns
        -------
        dict with budget status.
        """
        # Update usage from context
        if context:
            tokens = context.get("tokens_used", 0)
            cost = context.get("cost", tokens * self.token_cost_per_1k / 1000)

            self.tokens_used += tokens
            self.cost_incurred += cost

        issues = []
        over_budget = False

        # Check token budget
        if self.token_budget and self.tokens_used > self.token_budget:
            issues.append(f"Token budget exceeded: {self.tokens_used}/{self.token_budget}")
            over_budget = True

        # Check cost budget
        if self.cost_budget and self.cost_incurred > self.cost_budget:
            issues.append(
                f"Cost budget exceeded: ${self.cost_incurred:.4f}/${self.cost_budget:.4f}"
            )
            over_budget = True

        # Check time budget
        elapsed = time.time() - self.start_time
        if self.time_budget_seconds and elapsed > self.time_budget_seconds:
            issues.append(f"Time budget exceeded: {elapsed:.1f}s/{self.time_budget_seconds}s")
            over_budget = True

        # Calculate remaining budget
        remaining_tokens = (self.token_budget - self.tokens_used) if self.token_budget else None
        remaining_cost = (self.cost_budget - self.cost_incurred) if self.cost_budget else None
        remaining_time = (self.time_budget_seconds - elapsed) if self.time_budget_seconds else None

        return {
            "is_valid": not over_budget,
            "score": 1.0 if not over_budget else 0.0,
            "action": "block" if over_budget else "pass",
            "metadata": {
                "tokens_used": self.tokens_used,
                "tokens_remaining": remaining_tokens,
                "cost_incurred": round(self.cost_incurred, 4),
                "cost_remaining": round(remaining_cost, 4) if remaining_cost else None,
                "time_elapsed": round(elapsed, 1),
                "time_remaining": round(remaining_time, 1) if remaining_time else None,
                "issues": issues,
            },
        }

    def get_usage_report(self) -> Dict[str, Any]:
        """Get current usage report."""
        elapsed = time.time() - self.start_time

        return {
            "tokens": {
                "used": self.tokens_used,
                "budget": self.token_budget,
                "remaining": (self.token_budget - self.tokens_used) if self.token_budget else None,
                "utilization": self.tokens_used / self.token_budget if self.token_budget else None,
            },
            "cost": {
                "incurred": round(self.cost_incurred, 4),
                "budget": self.cost_budget,
                "remaining": round(self.cost_budget - self.cost_incurred, 4)
                if self.cost_budget
                else None,
                "utilization": self.cost_incurred / self.cost_budget if self.cost_budget else None,
            },
            "time": {
                "elapsed": round(elapsed, 1),
                "budget": self.time_budget_seconds,
                "remaining": round(self.time_budget_seconds - elapsed, 1)
                if self.time_budget_seconds
                else None,
            },
        }

    def reset(self):
        """Reset all counters."""
        self.tokens_used = 0
        self.cost_incurred = 0.0
        self.start_time = time.time()
