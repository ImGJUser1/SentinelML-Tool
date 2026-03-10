import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/rag/retrieval/relevance_scorer.py
"""
Retrieval relevance assessment for RAG.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseSentinelComponent


class RelevanceScorer(BaseSentinelComponent):
    """
    Score relevance between queries and retrieved documents.

    Uses multiple metrics:
    - Semantic similarity (embeddings)
    - Lexical overlap (BM25-style)
    - Query coverage (does retrieval cover all query aspects?)

    Parameters
    ----------
    embedding_model : callable, optional
        Function to compute embeddings.
    method : str, default='hybrid'
        Scoring method ('semantic', 'lexical', 'hybrid').

    Examples
    --------
    >>> scorer = RelevanceScorer()
    >>> scores = scorer.score(query, retrieved_docs)
    """

    def __init__(
        self,
        name: str = "RelevanceScorer",
        embedding_model: Optional[Any] = None,
        method: str = "hybrid",
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.embedding_model = embedding_model
        self.method = method

    def fit(self, X=None, y=None):
        """Fit (no-op for stateless scorer)."""
        self.is_fitted_ = True
        return self

    def score(
        self, query: str, documents: List[str], query_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Score relevance of documents to query.

        Returns
        -------
        dict with keys:
            - scores: list of individual scores
            - mean_score: average relevance
            - max_score: best document relevance
            - coverage: query aspect coverage
        """
        if not documents:
            return {"scores": [], "mean_score": 0.0, "max_score": 0.0, "coverage": 0.0}

        scores = []

        # Semantic similarity
        if self.method in ("semantic", "hybrid") and self.embedding_model:
            if query_embedding is None:
                query_emb = self.embedding_model.encode(query)
            else:
                query_emb = query_embedding

            for doc in documents:
                doc_emb = self.embedding_model.encode(doc)
                sim = self._cosine_sim(query_emb, doc_emb)
                scores.append(sim)

        # Lexical similarity
        if self.method in ("lexical", "hybrid"):
            lexical_scores = self._lexical_scores(query, documents)
            if scores:
                scores = [(s + l) / 2 for s, l in zip(scores, lexical_scores)]
            else:
                scores = lexical_scores

        # Compute coverage (are different aspects of query covered?)
        coverage = self._compute_coverage(query, documents)

        return {
            "scores": scores,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
            "coverage": coverage,
            "metadata": {"n_documents": len(documents), "method": self.method},
        }

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _lexical_scores(self, query: str, documents: List[str]) -> List[float]:
        """Compute lexical overlap scores."""
        query_terms = set(query.lower().split())
        scores = []
        for doc in documents:
            doc_terms = set(doc.lower().split())
            if not query_terms:
                scores.append(0.0)
                continue
            overlap = len(query_terms & doc_terms) / len(query_terms)
            scores.append(overlap)
        return scores

    def _compute_coverage(self, query: str, documents: List[str]) -> float:
        """Compute how well query aspects are covered."""
        # Simple version: check if query terms appear in any document
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0

        covered = set()
        for doc in documents:
            doc_terms = set(doc.lower().split())
            covered.update(query_terms & doc_terms)

        return len(covered) / len(query_terms)
