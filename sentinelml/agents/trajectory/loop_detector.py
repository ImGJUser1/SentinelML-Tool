import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/agents/trajectory/loop_detector.py
"""
Detect infinite loops and cyclic behavior in agents.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from sentinelml.core.base import BaseGuardrail


class LoopDetector(BaseGuardrail):
    """
    Detect repetitive/cyclic patterns in agent behavior.

    Identifies when agent is stuck in loops or
    making no meaningful progress.

    Parameters
    ----------
    window_size : int, default=5
        Number of steps to check for repetition.
    similarity_threshold : float, default=0.9
        Threshold for detecting similar states.

    Examples
    --------
    >>> detector = LoopDetector(window_size=5)
    >>> for step in trajectory:
    ...     result = detector.validate(step)
    ...     if not result['is_valid']:
    ...         print("Loop detected!")
    """

    def __init__(
        self,
        name: str = "LoopDetector",
        fail_mode: str = "block",
        window_size: int = 5,
        similarity_threshold: float = 0.9,
        max_repetitions: int = 3,
        verbose: bool = False,
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.max_repetitions = max_repetitions
        self.state_history: deque = deque(maxlen=100)
        self.action_history: deque = deque(maxlen=100)

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def validate(self, step: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Check for loops in recent history.

        Parameters
        ----------
        step : dict
            Current step with 'state', 'action', etc.

        Returns
        -------
        dict with loop detection results.
        """
        # Extract state and action
        state = step.get("state", str(step.get("input", "")))
        action = step.get("action", "")

        # Check for exact repetition
        exact_repeats = self._count_exact_repeats(state, action)

        # Check for cyclic patterns
        cycle_detected = self._detect_cycle()

        # Check for stuck state (high similarity, different actions)
        stuck_state = self._detect_stuck_state(state)

        is_loop = exact_repeats >= self.max_repetitions or cycle_detected or stuck_state

        # Record
        self.state_history.append(state)
        self.action_history.append(action)

        return {
            "is_valid": not is_loop,
            "score": 0.0 if is_loop else 1.0,
            "action": "block" if is_loop else "pass",
            "metadata": {
                "exact_repeats": exact_repeats,
                "cycle_detected": cycle_detected,
                "stuck_state": stuck_state,
                "history_size": len(self.state_history),
            },
        }

    def _count_exact_repeats(self, state: str, action: str) -> int:
        """Count exact repetitions of state-action pair."""
        count = 0
        check_window = list(self.state_history)[-self.window_size * 2 :]
        check_actions = list(self.action_history)[-self.window_size * 2 :]

        for i, (s, a) in enumerate(zip(check_window, check_actions)):
            if s == state and a == action:
                count += 1

        return count

    def _detect_cycle(self) -> bool:
        """Detect cyclic patterns in recent history."""
        if len(self.state_history) < self.window_size * 2:
            return False

        recent = list(self.state_history)[-self.window_size :]
        previous = list(self.state_history)[-self.window_size * 2 : -self.window_size]

        # Check if recent sequence matches previous
        similarity = self._sequence_similarity(recent, previous)
        return similarity > self.similarity_threshold

    def _detect_stuck_state(self, current_state: str) -> bool:
        """Detect if agent is stuck in similar states."""
        if len(self.state_history) < self.window_size:
            return False

        recent_states = list(self.state_history)[-self.window_size :]
        similarities = [self._state_similarity(current_state, s) for s in recent_states]

        # High average similarity = stuck
        avg_sim = sum(similarities) / len(similarities) if similarities else 0
        return avg_sim > self.similarity_threshold

    def _sequence_similarity(self, seq1: List, seq2: List) -> float:
        """Compute similarity between two sequences."""
        if len(seq1) != len(seq2):
            return 0.0

        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)

    def _state_similarity(self, state1: str, state2: str) -> float:
        """Compute similarity between two states."""
        # Simple Jaccard on words
        words1 = set(str(state1).lower().split())
        words2 = set(str(state2).lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union

    def reset(self):
        """Clear history."""
        self.state_history.clear()
        self.action_history.clear()
