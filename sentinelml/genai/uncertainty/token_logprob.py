import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/genai/uncertainty/token_logprob.py
"""
Uncertainty from token-level log probabilities.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseTrustModel


class TokenLogProb(BaseTrustModel):
    """
    Uncertainty quantification using native token log-probabilities.

    Uses the model's own probability estimates to compute
    perplexity and confidence metrics.

    Parameters
    ----------
    get_logprobs_fn : callable
        Function that returns log probabilities for tokens.
    aggregation : str, default='mean'
        How to aggregate token probabilities ('mean', 'min', 'perplexity').

    Examples
    --------
    >>> estimator = TokenLogProb(get_logprobs_fn=openai_client.get_logprobs)
    >>> confidence = estimator.score("Generate this text")
    """

    def __init__(
        self,
        get_logprobs_fn: Callable[[str], List[float]],
        name: str = "TokenLogProb",
        calibration_method: str = "isotonic",
        aggregation: str = "mean",
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.get_logprobs_fn = get_logprobs_fn
        self.aggregation = aggregation

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def score(self, X: str) -> np.float64:
        """
        Compute confidence from token log-probabilities.

        Returns
        -------
        score : float
            Trust score based on token probabilities.
        """
        if isinstance(X, (list, np.ndarray)):
            return np.array([self._score_single(x) for x in X])
        return self._score_single(X)

    def _score_single(self, text: str) -> np.float64:
        """Compute score for single text."""
        try:
            logprobs = self.get_logprobs_fn(text)
        except Exception as e:
            return np.float64(0.5)

        if not logprobs:
            return np.float64(0.5)

        logprobs = np.array(logprobs)
        probs = np.exp(logprobs)

        if self.aggregation == "mean":
            # Average token probability
            score = np.mean(probs)
        elif self.aggregation == "min":
            # Minimum token probability (most uncertain token)
            score = np.min(probs)
        elif self.aggregation == "perplexity":
            # Perplexity-based (normalized)
            perplexity = np.exp(-np.mean(logprobs))
            # Convert to [0, 1] score (lower perplexity = higher score)
            score = 1 / (1 + np.log(perplexity + 1))
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return np.float64(np.clip(score, 0, 1))

    def get_token_uncertainties(self, text: str) -> List[Dict[str, Any]]:
        """Get per-token uncertainty for visualization."""
        logprobs = self.get_logprobs_fn(text)
        tokens = text.split()  # Simplified tokenization

        uncertainties = []
        for i, (token, logprob) in enumerate(zip(tokens, logprobs)):
            uncertainties.append(
                {
                    "token": token,
                    "logprob": float(logprob),
                    "prob": float(np.exp(logprob)),
                    "position": i,
                }
            )

        return uncertainties
