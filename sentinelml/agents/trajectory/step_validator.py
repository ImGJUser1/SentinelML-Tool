import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/agents/trajectory/step_validator.py
"""
Per-step validation for agent trajectories.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseGuardrail


class StepValidator(BaseGuardrail):
    """
    Validate each step in an agent's trajectory.

    Checks that each action is valid, safe, and
    makes progress toward the goal.

    Parameters
    ----------
    valid_actions : list, optional
        List of valid action types.
    progress_checker : callable, optional
        Function to check if step makes progress.

    Examples
    --------
    >>> validator = StepValidator(valid_actions=['search', 'calc', 'retrieve'])
    >>> for step in trajectory:
    ...     result = validator.validate(step, context={'goal': goal})
    """

    def __init__(
        self,
        name: str = "StepValidator",
        fail_mode: str = "flag",
        valid_actions: Optional[List[str]] = None,
        progress_checker: Optional[Callable] = None,
        max_steps: int = 50,
        verbose: bool = False,
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)
        self.valid_actions = valid_actions or []
        self.progress_checker = progress_checker
        self.max_steps = max_steps
        self.step_history: List[Dict] = []

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def validate(self, step: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate a single step.

        Parameters
        ----------
        step : dict
            Step information with 'action', 'input', 'output', etc.
        context : dict, optional
            Should contain 'goal', 'previous_steps', etc.

        Returns
        -------
        dict with validation results.
        """
        issues = []
        score = 1.0

        # Check action validity
        if self.valid_actions:
            action = step.get("action", "")
            if action not in self.valid_actions:
                issues.append(f"Invalid action: {action}")
                score -= 0.3

        # Check for empty or malformed step
        if not step.get("input") and not step.get("output"):
            issues.append("Empty step")
            score -= 0.2

        # Check progress if checker provided
        if self.progress_checker and context:
            is_progress = self.progress_checker(step, context)
            if not is_progress:
                issues.append("No progress toward goal")
                score -= 0.2

        # Check for repetition
        if self._is_repetitive(step):
            issues.append("Repetitive step")
            score -= 0.3

        # Check step limit
        step_num = len(self.step_history)
        if step_num >= self.max_steps:
            issues.append(f"Exceeded max steps ({self.max_steps})")
            score = 0.0

        # Record step
        self.step_history.append({"step": step, "valid": len(issues) == 0, "score": score})

        is_valid = len(issues) == 0 and score > 0.5

        return {
            "is_valid": is_valid,
            "score": max(0, score),
            "action": "pass" if is_valid else self.fail_mode,
            "metadata": {
                "step_number": step_num,
                "issues": issues,
                "action_type": step.get("action", "unknown"),
            },
        }

    def _is_repetitive(self, step: Dict) -> bool:
        """Check if step repeats recent history."""
        if len(self.step_history) < 3:
            return False

        recent = self.step_history[-3:]
        current_str = str(step.get("input", "")) + str(step.get("action", ""))

        for prev in recent:
            prev_str = str(prev["step"].get("input", "")) + str(prev["step"].get("action", ""))
            if current_str == prev_str:
                return True

        return False

    def reset(self):
        """Clear step history."""
        self.step_history = []

    def get_trajectory_report(self) -> Dict[str, Any]:
        """Get report on full trajectory."""
        if not self.step_history:
            return {"message": "No steps recorded"}

        scores = [s["score"] for s in self.step_history]

        return {
            "total_steps": len(self.step_history),
            "valid_steps": sum(1 for s in self.step_history if s["valid"]),
            "average_score": np.mean(scores),
            "min_score": np.min(scores),
            "success_rate": sum(1 for s in self.step_history if s["valid"])
            / len(self.step_history),
        }
