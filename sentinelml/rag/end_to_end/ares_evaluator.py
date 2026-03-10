import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/rag/end_to_end/ares_evaluator.py
"""
ARES (Automated RAG Evaluation System) integration.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseSentinelComponent


class ARESEvaluator(BaseSentinelComponent):
    """
    ARES for synthetic RAG evaluation data generation.

    Generates synthetic questions and answers from documents
    for evaluation when ground truth is unavailable.

    Parameters
    ----------
    llm_client : callable
        LLM for synthetic data generation.
    n_samples : int, default=100
        Number of synthetic samples to generate.

    Examples
    --------
    >>> ares = ARESEvaluator(llm_client=openai_client)
    >>> synthetic_data = ares.generate(documents, n_samples=50)
    >>> ares.fit(synthetic_data)
    >>> scores = ares.evaluate(query, answer, contexts)
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        name: str = "ARESEvaluator",
        n_samples: int = 100,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.llm_client = llm_client
        self.n_samples = n_samples
        self.synthetic_data_ = []

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def generate(
        self, documents: List[str], n_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic evaluation data.

        Returns
        -------
        list of dicts with 'query', 'answer', 'context'.
        """
        if self.llm_client is None:
            raise ValueError("LLM client required for synthetic generation")

        n = n_samples or self.n_samples
        synthetic_data = []

        # Sample documents
        import random

        sampled_docs = random.choices(documents, k=min(n, len(documents)))

        for doc in sampled_docs:
            try:
                # Generate question from document
                prompt = f"""
                Based on the following document, generate a factual question
                that can be answered using only the information in the document.

                Document: {doc[:1000]}

                Generate a question and its answer in this format:
                Question: [question]
                Answer: [answer]
                """

                response = self.llm_client(prompt)

                # Parse response
                question, answer = self._parse_qa_response(response)

                if question and answer:
                    synthetic_data.append(
                        {"query": question, "answer": answer, "context": doc, "synthetic": True}
                    )

                if len(synthetic_data) >= n:
                    break

            except Exception as e:
                if self.verbose:
                    print(f"Generation failed: {e}")
                continue

        self.synthetic_data_ = synthetic_data
        return synthetic_data

    def _parse_qa_response(self, response: str) -> tuple:
        """Parse question and answer from LLM response."""
        question = None
        answer = None

        lines = response.strip().split("\n")
        for line in lines:
            if line.lower().startswith("question:"):
                question = line.split(":", 1)[1].strip()
            elif line.lower().startswith("answer:"):
                answer = line.split(":", 1)[1].strip()

        return question, answer

    def evaluate(self, query: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Evaluate using synthetic data as reference.

        Compares generated answer against synthetic examples
        to estimate quality.
        """
        if not self.synthetic_data_:
            return {"ares_score": 0.5, "note": "No synthetic data available. Run generate() first."}

        # Find most similar synthetic example
        similarities = []
        for item in self.synthetic_data_:
            # Simple similarity based on query overlap
            sim = self._query_similarity(query, item["query"])
            similarities.append((sim, item))

        # Get top match
        similarities.sort(reverse=True)
        best_match = similarities[0][1]

        # Compare answers
        answer_similarity = self._answer_similarity(answer, best_match["answer"])

        # Check context coverage
        context_score = self._context_coverage(query, contexts)

        return {
            "ares_score": (answer_similarity + context_score) / 2,
            "answer_similarity_to_synthetic": answer_similarity,
            "context_coverage": context_score,
            "matched_synthetic_query": best_match["query"],
            "synthetic_reference_answer": best_match["answer"],
        }

    def _query_similarity(self, q1: str, q2: str) -> float:
        """Compute similarity between queries."""
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union

    def _answer_similarity(self, a1: str, a2: str) -> float:
        """Compute similarity between answers."""
        words1 = set(a1.lower().split())
        words2 = set(a2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union

    def _context_coverage(self, query: str, contexts: List[str]) -> float:
        """Check if contexts cover query aspects."""
        query_words = set(query.lower().split())

        if not query_words:
            return 1.0

        covered = set()
        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            covered.update(query_words & ctx_words)

        return len(covered) / len(query_words)
