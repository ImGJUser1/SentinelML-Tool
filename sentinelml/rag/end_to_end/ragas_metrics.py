from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/rag/end_to_end/ragas_metrics.py
"""
RAGAS (Retrieval-Augmented Generation Assessment) integration.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseSentinelComponent


class RAGASEvaluator(BaseSentinelComponent):
    """
    RAGAS metrics for end-to-end RAG evaluation.

    Implements standard RAGAS metrics:
    - Faithfulness: Answer grounded in retrieved context
    - Answer Relevancy: Answer addresses question
    - Context Precision: Relevant contexts retrieved
    - Context Recall: All relevant contexts retrieved

    Parameters
    ----------
    metrics : list, default=['faithfulness', 'answer_relevancy']
        Which RAGAS metrics to compute.

    Examples
    --------
    >>> evaluator = RAGASEvaluator(metrics=['faithfulness', 'context_recall'])
    >>> scores = evaluator.evaluate(query, answer, contexts)
    """

    AVAILABLE_METRICS = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "context_entity_recall",
        "answer_similarity",
        "answer_correctness",
    ]

    def __init__(
        self,
        name: str = "RAGASEvaluator",
        metrics: Optional[List[str]] = None,
        embedding_model: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.metrics = metrics or ["faithfulness", "answer_relevancy"]
        self.embedding_model = embedding_model
        self.llm_client = llm_client

        # Validate metrics
        invalid = [m for m in self.metrics if m not in self.AVAILABLE_METRICS]
        if invalid:
            raise ValueError(f"Invalid metrics: {invalid}")

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def evaluate(
        self, query: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute RAGAS metrics.

        Returns
        -------
        dict with all requested metric scores.
        """
        results = {}

        if "faithfulness" in self.metrics:
            results["faithfulness"] = self._compute_faithfulness(answer, contexts)

        if "answer_relevancy" in self.metrics:
            results["answer_relevancy"] = self._compute_answer_relevancy(query, answer)

        if "context_precision" in self.metrics:
            results["context_precision"] = self._compute_context_precision(query, contexts)

        if "context_recall" in self.metrics:
            results["context_recall"] = self._compute_context_recall(query, contexts)

        if "answer_correctness" in self.metrics and ground_truth:
            results["answer_correctness"] = self._compute_answer_correctness(answer, ground_truth)

        # Overall score
        results["overall"] = np.mean([v for v in results.values() if isinstance(v, (int, float))])

        return results

    def _compute_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Compute faithfulness metric."""
        # Extract claims from answer
        claims = self._extract_claims(answer)

        if not claims:
            return 1.0

        # Check each claim against contexts
        supported = 0
        for claim in claims:
            for ctx in contexts:
                if self._claim_supported_by_context(claim, ctx):
                    supported += 1
                    break

        return supported / len(claims)

    def _compute_answer_relevancy(self, query: str, answer: str) -> float:
        """Compute answer relevancy."""
        if self.embedding_model is None:
            # Fallback: keyword overlap
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(query_words & answer_words) / len(query_words) if query_words else 0
            return overlap

        query_emb = self.embedding_model.encode(query)
        answer_emb = self.embedding_model.encode(answer)

        similarity = float(
            np.dot(query_emb, answer_emb)
            / (np.linalg.norm(query_emb) * np.linalg.norm(answer_emb) + 1e-8)
        )

        return similarity

    def _compute_context_precision(self, query: str, contexts: List[str]) -> float:
        """Compute context precision."""
        if not contexts:
            return 0.0

        # Check how many contexts are relevant to query
        relevant = 0
        for ctx in contexts:
            if self._is_relevant(query, ctx):
                relevant += 1

        return relevant / len(contexts)

    def _compute_context_recall(self, query: str, contexts: List[str]) -> float:
        """Compute context recall."""
        # Extract query aspects
        aspects = self._extract_aspects(query)

        if not aspects:
            return 1.0

        # Check how many aspects are covered by contexts
        covered = 0
        for aspect in aspects:
            for ctx in contexts:
                if self._aspect_in_context(aspect, ctx):
                    covered += 1
                    break

        return covered / len(aspects)

    def _compute_answer_correctness(self, answer: str, ground_truth: str) -> float:
        """Compute answer correctness against ground truth."""
        if self.embedding_model is None:
            # Exact match fallback
            return 1.0 if answer.strip().lower() == ground_truth.strip().lower() else 0.0

        answer_emb = self.embedding_model.encode(answer)
        truth_emb = self.embedding_model.encode(ground_truth)

        similarity = float(
            np.dot(answer_emb, truth_emb)
            / (np.linalg.norm(answer_emb) * np.linalg.norm(truth_emb) + 1e-8)
        )

        return similarity

    def _extract_claims(self, answer: str) -> List[str]:
        """Extract factual claims from answer."""
        sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
        # Filter for factual statements
        factual = []
        for sent in sentences:
            if any(w in sent.lower().split() for w in ["is", "are", "was", "were", "has", "have"]):
                factual.append(sent)
        return factual[:15]

    def _claim_supported_by_context(self, claim: str, context: str) -> bool:
        """Check if claim is supported by context."""
        if self.embedding_model is not None:
            claim_emb = self.embedding_model.encode(claim)
            ctx_emb = self.embedding_model.encode(context)

            similarity = float(
                np.dot(claim_emb, ctx_emb)
                / (np.linalg.norm(claim_emb) * np.linalg.norm(ctx_emb) + 1e-8)
            )

            return similarity > 0.7

        # Keyword fallback
        claim_words = set(claim.lower().split())
        ctx_words = set(context.lower().split())
        overlap = len(claim_words & ctx_words) / len(claim_words) if claim_words else 0
        return overlap > 0.5

    def _is_relevant(self, query: str, context: str) -> bool:
        """Check if context is relevant to query."""
        if self.embedding_model is not None:
            query_emb = self.embedding_model.encode(query)
            ctx_emb = self.embedding_model.encode(context)

            similarity = float(
                np.dot(query_emb, ctx_emb)
                / (np.linalg.norm(query_emb) * np.linalg.norm(ctx_emb) + 1e-8)
            )

            return similarity > 0.6

        # Keyword fallback
        query_words = set(query.lower().split())
        ctx_words = set(context.lower().split())
        overlap = len(query_words & ctx_words) / len(query_words) if query_words else 0
        return overlap > 0.3

    def _extract_aspects(self, query: str) -> List[str]:
        """Extract key aspects from query."""
        # Simple noun phrase extraction
        import re

        # Find capitalized phrases
        aspects = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query)
        # Find quoted phrases
        aspects.extend(re.findall(r'"([^"]+)"', query))
        return list(set(aspects))[:10]

    def _aspect_in_context(self, aspect: str, context: str) -> bool:
        """Check if aspect appears in context."""
        return aspect.lower() in context.lower()
