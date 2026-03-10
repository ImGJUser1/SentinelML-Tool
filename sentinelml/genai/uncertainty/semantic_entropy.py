import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/genai/uncertainty/semantic_entropy.py
"""
Semantic entropy for detecting hallucinations in LLMs.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseTrustModel


class SemanticEntropy(BaseTrustModel):
    """
    Detect hallucinations using semantic entropy.

    High entropy over semantic meaning (rather than tokens)
    indicates uncertainty and potential hallucination.

    Based on: "Detecting Hallucinations in Large Language Models
    Using Semantic Entropy" (Farquhar et al., 2024)

    Parameters
    ----------
    generate_fn : callable
        Function to generate text from prompt.
    n_samples : int, default=10
        Number of generations for entropy estimation.
    clustering_threshold : float, default=0.9
        Similarity threshold for semantic clustering.

    Examples
    --------
    >>> estimator = SemanticEntropy(generate_fn=llm.generate)
    >>> estimator.fit(prompts_with_known_answers)
    >>> uncertainty = estimator.score(new_prompt)
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        name: str = "SemanticEntropy",
        calibration_method: str = "isotonic",
        n_samples: int = 10,
        clustering_threshold: float = 0.9,
        embedding_model: Optional[Any] = None,
        temperature: float = 1.0,
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.generate_fn = generate_fn
        self.n_samples = n_samples
        self.clustering_threshold = clustering_threshold
        self.embedding_model = embedding_model
        self.temperature = temperature

    def fit(self, X: List[str], y: Optional[List[str]] = None) -> "SemanticEntropy":
        """
        Fit on prompts with known answers.

        X : list of prompts
        y : list of expected answers (for calibration)
        """
        self.is_fitted_ = True
        return self

    def score(self, X: str) -> np.float64:
        """
        Compute semantic entropy for a prompt.

        Returns
        -------
        score : float
            Trust score (1 - normalized entropy).
        """
        if isinstance(X, (list, np.ndarray)):
            # Batch processing
            return np.array([self._score_single(x) for x in X])
        return self._score_single(X)

    def _score_single(self, prompt: str) -> np.float64:
        """Compute semantic entropy for single prompt."""
        # Generate multiple answers
        generations = []
        for _ in range(self.n_samples):
            try:
                answer = self.generate_fn(prompt)
                generations.append(answer)
            except Exception as e:
                continue

        if len(generations) < 3:
            return np.float64(0.5)  # Insufficient samples

        # Cluster by semantic similarity
        clusters = self._cluster_generations(generations)

        # Compute entropy over clusters
        cluster_probs = np.array([len(c) for c in clusters]) / len(generations)
        entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))

        # Normalize by max possible entropy
        max_entropy = np.log(len(generations))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Convert to trust score
        trust = 1 - normalized_entropy

        return np.float64(trust)

    def _cluster_generations(self, generations: List[str]) -> List[List[str]]:
        """Cluster generations by semantic similarity."""
        if self.embedding_model is None:
            # Fallback: exact match clustering
            clusters = []
            for gen in generations:
                found = False
                for cluster in clusters:
                    if self._text_similarity(gen, cluster[0]) > self.clustering_threshold:
                        cluster.append(gen)
                        found = True
                        break
                if not found:
                    clusters.append([gen])
            return clusters

        # Embedding-based clustering
        embeddings = [self.embedding_model.encode(g) for g in generations]

        # Simple greedy clustering
        clusters = []
        used = set()

        for i, emb in enumerate(embeddings):
            if i in used:
                continue

            cluster = [generations[i]]
            used.add(i)

            for j, other_emb in enumerate(embeddings[i + 1 :], i + 1):
                if j in used:
                    continue

                similarity = float(
                    np.dot(emb, other_emb)
                    / (np.linalg.norm(emb) * np.linalg.norm(other_emb) + 1e-8)
                )

                if similarity > self.clustering_threshold:
                    cluster.append(generations[j])
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity without embeddings."""
        # Jaccard similarity on word sets
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def get_generations(self, prompt: str) -> List[str]:
        """Get all sampled generations for inspection."""
        generations = []
        for _ in range(self.n_samples):
            try:
                answer = self.generate_fn(prompt)
                generations.append(answer)
            except Exception:
                continue
        return generations
