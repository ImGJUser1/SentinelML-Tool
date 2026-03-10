import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/rag/retrieval/diversity_metrics.py
"""
Diversity metrics for retrieved documents.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseSentinelComponent


class DiversityMetrics(BaseSentinelComponent):
    """
    Measure diversity in retrieval results.

    Ensures retrieved documents provide varied perspectives
    and aren't redundant.

    Parameters
    ----------
    embedding_model : callable
        For semantic similarity calculation.
    metrics : list, default=['pairwise', 'cluster']
        Diversity metrics to compute.

    Examples
    --------
    >>> metrics = DiversityMetrics(embedding_model=encoder)
    >>> diversity = metrics.compute(documents)
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        name: str = "DiversityMetrics",
        metrics: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.embedding_model = embedding_model
        self.metrics = metrics or ["pairwise", "cluster"]

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def compute(self, documents: List[str]) -> Dict[str, Any]:
        """
        Compute diversity metrics for documents.

        Returns
        -------
        dict with diversity scores.
        """
        if len(documents) < 2:
            return {
                "diversity_score": 0.0,
                "pairwise_dissimilarity": 0.0,
                "cluster_diversity": 0.0,
                "redundancy_detected": False,
            }

        # Get embeddings
        if self.embedding_model is not None:
            embeddings = np.array([self.embedding_model.encode(d) for d in documents])
        else:
            # Bag-of-words fallback
            embeddings = self._bow_embeddings(documents)

        results = {}

        # Pairwise dissimilarity
        if "pairwise" in self.metrics:
            pairwise_sim = self._pairwise_similarity(embeddings)
            results["pairwise_dissimilarity"] = 1 - np.mean(pairwise_sim)

        # Cluster-based diversity
        if "cluster" in self.metrics:
            results["cluster_diversity"] = self._cluster_diversity(embeddings)

        # Topic coverage (if many documents on same topic = low diversity)
        if "topic" in self.metrics:
            results["topic_diversity"] = self._topic_diversity(documents)

        # Overall score
        scores = [v for k, v in results.items() if "diversity" in k or "dissimilarity" in k]
        results["diversity_score"] = np.mean(scores) if scores else 0.5

        # Detect redundancy
        results["redundancy_detected"] = results["diversity_score"] < 0.3

        # Identify redundant pairs
        if results["redundancy_detected"]:
            results["redundant_pairs"] = self._find_redundant_pairs(embeddings, documents)

        return results

    def _bow_embeddings(self, documents: List[str]) -> np.ndarray:
        """Create bag-of-words embeddings."""
        # Simple vocabulary
        vocab = set()
        for doc in documents:
            vocab.update(doc.lower().split())
        vocab = sorted(vocab)

        vectors = []
        for doc in documents:
            words = doc.lower().split()
            vec = [words.count(w) for w in vocab]
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)

        return np.array(vectors)

    def _pairwise_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarities."""
        n = len(embeddings)
        similarities = []

        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)

        return np.array(similarities)

    def _cluster_diversity(self, embeddings: np.ndarray) -> float:
        """Measure diversity via clustering."""
        from sklearn.cluster import KMeans

        n = len(embeddings)
        if n < 3:
            return 0.0

        # Determine optimal k
        k = min(int(np.sqrt(n)), n // 2, 10)
        if k < 2:
            return 0.0

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Diversity = entropy of cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(k)

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _topic_diversity(self, documents: List[str]) -> float:
        """Simple topic diversity based on keyword overlap."""
        # Extract key terms from each document
        doc_terms = []
        for doc in documents:
            words = set(doc.lower().split())
            # Remove common stop words
            stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been"}
            words = words - stop_words
            doc_terms.append(words)

        # Compute pairwise Jaccard dissimilarity
        dissimilarities = []
        for i in range(len(doc_terms)):
            for j in range(i + 1, len(doc_terms)):
                intersection = len(doc_terms[i] & doc_terms[j])
                union = len(doc_terms[i] | doc_terms[j])
                jaccard = intersection / union if union > 0 else 0
                dissimilarities.append(1 - jaccard)

        return np.mean(dissimilarities) if dissimilarities else 0.0

    def _find_redundant_pairs(
        self, embeddings: np.ndarray, documents: List[str], threshold: float = 0.9
    ) -> List[Dict]:
        """Find pairs of highly similar documents."""
        redundant = []
        n = len(embeddings)

        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j])
                if sim > threshold:
                    redundant.append(
                        {
                            "doc_1_idx": i,
                            "doc_2_idx": j,
                            "similarity": float(sim),
                            "doc_1_preview": documents[i][:100] + "...",
                            "doc_2_preview": documents[j][:100] + "...",
                        }
                    )

        return redundant
