import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/rag/generation/answer_relevance.py
"""
Answer relevance scoring for RAG.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseSentinelComponent


class AnswerRelevance(BaseSentinelComponent):
    """
    Score how relevant an answer is to the original query.

    Different from faithfulness—measures if answer addresses
    what was asked, not if it's factually correct.

    Parameters
    ----------
    embedding_model : callable, optional
        For semantic similarity.
    method : str, default='direct'
        Scoring method ('direct', 'qa_cross_encoder', 'keyword').
    threshold : float, default=0.6
        Minimum score to consider answer relevant.
    cross_encoder : callable, optional
        Cross-encoder model for QA relevance.
    name : str, default='AnswerRelevance'
        Component name.
    verbose : bool, default=False
        Enable verbose logging.

    Examples
    --------
    >>> scorer = AnswerRelevance(embedding_model=encoder)
    >>> relevance = scorer.score(query, answer)
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        method: str = "direct",
        threshold: float = 0.6,
        cross_encoder: Optional[Any] = None,
        name: str = "AnswerRelevance",
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.embedding_model = embedding_model
        self.method = method
        self.threshold = threshold
        self.cross_encoder = cross_encoder

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def score(self, query: str, answer: str) -> Dict[str, Any]:
        """
        Score relevance of answer to query.

        Parameters
        ----------
        query : str
            Original user query.
        answer : str
            Generated answer to evaluate.

        Returns
        -------
        dict with relevance score and analysis.
        """
        if self.method == "direct":
            relevance = self._direct_similarity(query, answer)
        elif self.method == "qa_cross_encoder":
            relevance = self._cross_encoder_score(query, answer)
        elif self.method == "keyword":
            relevance = self._keyword_overlap(query, answer)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        is_relevant = relevance >= self.threshold

        return {
            "relevance_score": relevance,
            "is_relevant": is_relevant,
            "query_length": len(query.split()),
            "answer_length": len(answer.split()),
            "method": self.method,
            "threshold": self.threshold,
        }

    def _direct_similarity(self, query: str, answer: str) -> float:
        """Compute direct embedding similarity."""
        if self.embedding_model is None:
            raise ValueError("Embedding model required for direct similarity method")

        # Assuming encode method exists - adjust based on your embedding model
        query_emb = self.embedding_model.encode(query)
        answer_emb = self.embedding_model.encode(answer)

        # Convert to numpy arrays if needed
        query_emb = np.array(query_emb)
        answer_emb = np.array(answer_emb)

        # Compute cosine similarity
        similarity = float(
            np.dot(query_emb, answer_emb)
            / (np.linalg.norm(query_emb) * np.linalg.norm(answer_emb) + 1e-8)
        )

        return similarity

    def _cross_encoder_score(self, query: str, answer: str) -> float:
        """Use cross-encoder for QA relevance."""
        if self.cross_encoder is None:
            raise ValueError("Cross-encoder required for qa_cross_encoder method")

        # Cross-encoders take pairs and output relevance score
        # Adjust based on your cross-encoder interface
        score = self.cross_encoder.predict([(query, answer)])[0]
        return float(score)

    def _keyword_overlap(self, query: str, answer: str) -> float:
        """Simple keyword overlap as fallback."""
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())

        # Remove stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "to",
            "for",
            "of",
            "in",
        }
        query_terms = query_terms - stop_words
        answer_terms = answer_terms - stop_words

        if not query_terms:
            return 0.0

        overlap = len(query_terms & answer_terms) / len(query_terms)
        return overlap

    def score_batch(self, queries: List[str], answers: List[str]) -> List[Dict[str, Any]]:
        """
        Score relevance for multiple query-answer pairs.

        Parameters
        ----------
        queries : List[str]
            List of queries.
        answers : List[str]
            List of answers (must match length of queries).

        Returns
        -------
        List of relevance score dictionaries.
        """
        if len(queries) != len(answers):
            raise ValueError(
                f"Number of queries ({len(queries)}) must match number of answers ({len(answers)})"
            )

        results = []
        for query, answer in zip(queries, answers):
            results.append(self.score(query, answer))

        return results
