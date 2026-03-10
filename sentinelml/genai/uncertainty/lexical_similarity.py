import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/genai/uncertainty/lexical_similarity.py
"""
Lexical similarity-based uncertainty for LLMs.
"""

from collections import Counter
from typing import Callable, List, Optional

import numpy as np

from sentinelml.core.base import BaseTrustModel


class LexicalSimilarity(BaseTrustModel):
    """
    Uncertainty based on lexical variation across generations.

    Measures diversity in token choices, n-gram overlap,
    and syntactic structures across multiple samples.

    Parameters
    ----------
    generate_fn : callable
        Text generation function.
    n_samples : int, default=5
        Number of generations.
    metrics : list, default=['rouge', 'bleu', 'edit_distance']
        Similarity metrics to compute.

    Examples
    --------
    >>> estimator = LexicalSimilarity(generate_fn=llm.generate, n_samples=10)
    >>> consistency = estimator.score("What is 2+2?")
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        name: str = "LexicalSimilarity",
        calibration_method: str = "isotonic",
        n_samples: int = 5,
        metrics: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.generate_fn = generate_fn
        self.n_samples = n_samples
        self.metrics = metrics or ["rouge", "bleu", "edit_distance"]

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def score(self, X: str) -> np.float64:
        """
        Compute lexical consistency score.

        Returns
        -------
        score : float
            Trust based on lexical consistency (higher = more consistent).
        """
        if isinstance(X, (list, np.ndarray)):
            return np.array([self._score_single(x) for x in X])
        return self._score_single(X)

    def _score_single(self, prompt: str) -> np.float64:
        """Compute consistency for single prompt."""
        # Generate multiple answers
        generations = []
        for _ in range(self.n_samples):
            try:
                answer = self.generate_fn(prompt)
                generations.append(answer)
            except Exception:
                continue

        if len(generations) < 2:
            return np.float64(0.5)

        # Compute pairwise similarities
        similarities = []
        for i, gen1 in enumerate(generations):
            for gen2 in generations[i + 1 :]:
                sim = self._compute_similarity(gen1, gen2)
                similarities.append(sim)

        avg_similarity = np.mean(similarities)
        return np.float64(avg_similarity)

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity using selected metrics."""
        scores = []

        if "rouge" in self.metrics:
            scores.append(self._rouge_l(text1, text2))

        if "bleu" in self.metrics:
            scores.append(self._bleu(text1, text2))

        if "edit_distance" in self.metrics:
            scores.append(self._normalized_edit_distance(text1, text2))

        if "jaccard" in self.metrics:
            scores.append(self._jaccard(text1, text2))

        return np.mean(scores) if scores else 0.5

    def _rouge_l(self, text1: str, text2: str) -> float:
        """Compute ROUGE-L (Longest Common Subsequence)."""
        words1 = text1.lower().split()
        words2 = text2.lower().split()

        # LCS length
        lcs_length = self._lcs_length(words1, words2)

        if not words1 or not words2:
            return 0.0

        precision = lcs_length / len(words1)
        recall = lcs_length / len(words2)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute Longest Common Subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _bleu(self, text1: str, text2: str) -> float:
        """Simplified BLEU score."""
        # Use 1-gram precision as approximation
        words1 = text1.lower().split()
        words2 = text2.lower().split()

        if not words1 or not words2:
            return 0.0

        counter1 = Counter(words1)
        counter2 = Counter(words2)

        overlap = sum((counter1 & counter2).values())
        precision = overlap / len(words1)

        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(words2) / len(words1))) if len(words1) > 0 else 0

        return precision * bp

    def _normalized_edit_distance(self, text1: str, text2: str) -> float:
        """Compute normalized Levenshtein distance."""
        import difflib

        distance = difflib.SequenceMatcher(None, text1, text2).distance()
        max_len = max(len(text1), len(text2))

        if max_len == 0:
            return 1.0

        similarity = 1 - (distance / max_len)
        return max(0.0, similarity)

    def _jaccard(self, text1: str, text2: str) -> float:
        """Jaccard similarity on word sets."""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0
