import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/agents/reasoning/step_consistency.py
"""
Check consistency across reasoning steps.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseGuardrail


class StepConsistency(BaseGuardrail):
    """
    Verify consistency across chain-of-thought steps.

    Ensures each step follows logically from previous ones
    and conclusions match premises.

    Parameters
    ----------
    embedding_model : callable
        For semantic similarity checking.

    Examples
    --------
    >>> checker = StepConsistency(embedding_model=encoder)
    >>> steps = ["Step 1...", "Step 2...", "Conclusion..."]
    >>> result = checker.validate(steps)
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        name: str = "StepConsistency",
        fail_mode: str = "flag",
        coherence_threshold: float = 0.6,
        verbose: bool = False,
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)
        self.embedding_model = embedding_model
        self.coherence_threshold = coherence_threshold

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def validate(self, steps: List[str], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Check consistency of reasoning steps.

        Parameters
        ----------
        steps : list of str
            Ordered reasoning steps.

        Returns
        -------
        dict with consistency analysis.
        """
        if len(steps) < 2:
            return {"is_valid": True, "score": 1.0, "coherence_scores": [], "breakpoints": []}

        coherence_scores = []
        breakpoints = []

        # Check coherence between consecutive steps
        for i in range(len(steps) - 1):
            score = self._step_coherence(steps[i], steps[i + 1])
            coherence_scores.append(score)

            if score < self.coherence_threshold:
                breakpoints.append(
                    {
                        "between_steps": (i, i + 1),
                        "coherence": score,
                        "step_1_preview": steps[i][:100],
                        "step_2_preview": steps[i + 1][:100],
                    }
                )

        # Check conclusion follows from premises
        if len(steps) > 2:
            premises = " ".join(steps[:-1])
            conclusion = steps[-1]
            conclusion_coherence = self._step_coherence(premises, conclusion)

            if conclusion_coherence < self.coherence_threshold:
                breakpoints.append(
                    {"type": "conclusion_mismatch", "coherence": conclusion_coherence}
                )

        avg_coherence = np.mean(coherence_scores) if coherence_scores else 1.0
        is_consistent = len(breakpoints) == 0 and avg_coherence > self.coherence_threshold

        return {
            "is_valid": is_consistent,
            "score": avg_coherence,
            "action": "pass" if is_consistent else self.fail_mode,
            "metadata": {
                "coherence_scores": coherence_scores,
                "average_coherence": avg_coherence,
                "breakpoints": breakpoints,
                "n_steps": len(steps),
            },
        }

    def _step_coherence(self, step1: str, step2: str) -> float:
        """Measure semantic coherence between steps."""
        if self.embedding_model is not None:
            emb1 = self.embedding_model.encode(step1)
            emb2 = self.embedding_model.encode(step2)

            similarity = float(
                np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            )
            return similarity

        # Keyword overlap fallback
        words1 = set(step1.lower().split())
        words2 = set(step2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Check for shared terms (excluding stop words)
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "but"}
        content1 = words1 - stop_words
        content2 = words2 - stop_words

        if not content1:
            return 1.0  # Empty step, assume coherent

        overlap = len(content1 & content2) / len(content1)
        return overlap
