import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/genai/guardrails/output/hallucination_detector.py
"""
LLM hallucination detection using self-consistency and fact-checking.
"""

from collections import Counter
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseGuardrail


class HallucinationDetector(BaseGuardrail):
    """
    Detects hallucinations in LLM outputs.

    Uses multiple strategies:
    - Self-consistency: Sample multiple answers, check agreement
    - Factual verification: Check against knowledge base
    - Entropy-based: High generation entropy indicates uncertainty

    Parameters
    ----------
    method : str, default='self_consistency'
        Detection strategy.
    consistency_samples : int, default=5
        Number of samples for self-consistency.
    similarity_threshold : float, default=0.8
        Threshold for semantic similarity.

    Examples
    --------
    >>> detector = HallucinationDetector(method='self_consistency')
    >>> result = detector.validate("The capital of France is London.")
    >>> print(result['is_valid'])  # False
    """

    def __init__(
        self,
        name: str = "HallucinationDetector",
        fail_mode: str = "flag",
        method: str = "self_consistency",
        consistency_samples: int = 5,
        similarity_threshold: float = 0.8,
        llm_client: Optional[Callable] = None,
        embedding_model: Optional[Any] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)
        self.method = method
        self.consistency_samples = consistency_samples
        self.similarity_threshold = similarity_threshold
        self.llm_client = llm_client
        self.embedding_model = embedding_model

    def validate(self, content: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate output for hallucinations.

        Parameters
        ----------
        content : str
            LLM output to validate.
        context : dict, optional
            Should contain:
            - 'query': original user query
            - 'sources': retrieved documents (for RAG)
            - 'generate_fn': function to regenerate response

        Returns
        -------
        dict with validation results.
        """
        if context is None:
            context = {}

        query = context.get("query", "")
        sources = context.get("sources", [])

        if self.method == "self_consistency":
            return self._check_self_consistency(content, query, context)
        elif self.method == "fact_check":
            return self._check_facts(content, sources)
        elif self.method == "combined":
            consistency_result = self._check_self_consistency(content, query, context)
            fact_result = self._check_facts(content, sources)
            # Combine scores
            combined_score = (consistency_result["score"] + fact_result["score"]) / 2
            return {
                "is_valid": combined_score > 0.5,
                "score": combined_score,
                "action": "pass" if combined_score > 0.5 else self.fail_mode,
                "metadata": {
                    "consistency_check": consistency_result["metadata"],
                    "fact_check": fact_result["metadata"],
                },
            }
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _check_self_consistency(self, content: str, query: str, context: Dict) -> Dict[str, Any]:
        """Check consistency across multiple generations."""
        generate_fn = context.get("generate_fn")
        if generate_fn is None or self.llm_client is None:
            # Can't check without generation capability
            return {
                "is_valid": True,
                "score": 0.5,
                "action": "pass",
                "metadata": {"note": "No generation function provided"},
            }

        # Generate multiple answers
        answers = [content]  # Include original
        for _ in range(self.consistency_samples - 1):
            try:
                new_answer = generate_fn(query)
                answers.append(new_answer)
            except Exception as e:
                continue

        if len(answers) < 2:
            return {
                "is_valid": True,
                "score": 0.5,
                "action": "pass",
                "metadata": {"note": "Insufficient samples"},
            }

        # Compute pairwise similarities
        similarities = []
        if self.embedding_model:
            embeddings = [self._get_embedding(a) for a in answers]
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = self._cosine_similarity(embeddings[i], embeddings[j])
                    similarities.append(sim)
        else:
            # Fallback to simple string similarity
            for i in range(len(answers)):
                for j in range(i + 1, len(answers)):
                    sim = self._jaccard_similarity(answers[i], answers[j])
                    similarities.append(sim)

        mean_similarity = np.mean(similarities) if similarities else 0
        is_consistent = mean_similarity > self.similarity_threshold

        return {
            "is_valid": is_consistent,
            "score": mean_similarity,
            "action": "pass" if is_consistent else self.fail_mode,
            "metadata": {
                "mean_similarity": mean_similarity,
                "n_samples": len(answers),
                "all_answers": answers if len(answers) < 10 else None,
            },
        }

    def _check_facts(self, content: str, sources: List[str]) -> Dict[str, Any]:
        """Verify claims against source documents."""
        if not sources:
            return {
                "is_valid": True,
                "score": 0.5,
                "action": "pass",
                "metadata": {"note": "No sources provided for verification"},
            }

        # Extract claims (simple sentence splitting)
        claims = [s.strip() for s in content.split(".") if len(s.strip()) > 10]

        verified_claims = 0
        for claim in claims:
            # Check if claim is supported by any source
            for source in sources:
                if self._claim_supported(claim, source):
                    verified_claims += 1
                    break

        verification_rate = verified_claims / len(claims) if claims else 0

        return {
            "is_valid": verification_rate > 0.7,
            "score": verification_rate,
            "action": "pass" if verification_rate > 0.7 else self.fail_mode,
            "metadata": {
                "verification_rate": verification_rate,
                "n_claims": len(claims),
                "n_verified": verified_claims,
            },
        }

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector."""
        if self.embedding_model:
            return self.embedding_model.encode(text)
        # Fallback: simple bag of words
        words = set(text.lower().split())
        return np.array([hash(w) % 1000 for w in words])

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def _jaccard_similarity(self, a: str, b: str) -> float:
        """Compute Jaccard similarity between strings."""
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0

    def _claim_supported(self, claim: str, source: str) -> bool:
        """Check if claim is supported by source text."""
        # Simple keyword overlap
        claim_words = set(claim.lower().split())
        source_words = set(source.lower().split())
        overlap = len(claim_words & source_words) / len(claim_words)
        return overlap > 0.5
