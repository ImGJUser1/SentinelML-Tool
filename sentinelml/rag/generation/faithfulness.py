import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/rag/generation/faithfulness.py
"""
Faithfulness checking for RAG-generated answers.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseSentinelComponent


class FaithfulnessChecker(BaseSentinelComponent):
    """
    Verify that answers are faithful to retrieved contexts.

    Detects hallucinations and unsupported claims by
    comparing answer against source documents.

    Parameters
    ----------
    embedding_model : callable
        For semantic comparison.
    method : str, default='claim_extraction'
        Faithfulness checking method.

    Examples
    --------
    >>> checker = FaithfulnessChecker(embedding_model=encoder)
    >>> result = checker.check(answer, contexts)
    >>> print(result['faithfulness_score'])
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        name: str = "FaithfulnessChecker",
        method: str = "claim_extraction",
        threshold: float = 0.7,
        nli_model: Optional[Any] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.embedding_model = embedding_model
        self.method = method
        self.threshold = threshold
        self.nli_model = nli_model  # Natural Language Inference model

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def check(self, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Check faithfulness of answer to contexts.

        Returns
        -------
        dict with faithfulness analysis.
        """
        # Extract claims from answer
        claims = self._extract_claims(answer)

        if not claims:
            return {
                "faithfulness_score": 1.0,
                "claims_checked": 0,
                "supported_claims": [],
                "unsupported_claims": [],
                "verification_method": self.method,
            }

        supported = []
        unsupported = []

        for claim in claims:
            is_supported, evidence = self._verify_claim(claim, contexts)
            if is_supported:
                supported.append({"claim": claim, "evidence": evidence})
            else:
                unsupported.append({"claim": claim, "reason": "No supporting evidence found"})

        # Calculate score
        if len(claims) > 0:
            faithfulness = len(supported) / len(claims)
        else:
            faithfulness = 1.0

        return {
            "faithfulness_score": faithfulness,
            "claims_checked": len(claims),
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "is_faithful": faithfulness >= self.threshold,
            "verification_method": self.method,
        }

    def _extract_claims(self, answer: str) -> List[str]:
        """Extract factual claims from answer."""
        # Simple sentence splitting
        sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]

        # Filter for factual statements (heuristic)
        factual_indicators = ["is", "are", "was", "were", "has", "have", "does", "do"]
        claims = []

        for sent in sentences:
            # Check if sentence contains factual indicators
            if any(ind in sent.lower().split() for ind in factual_indicators):
                # Skip questions and conditionals
                if not sent.endswith("?") and "if" not in sent.lower()[:10]:
                    claims.append(sent)

        return claims[:20]  # Limit to 20 claims

    def _verify_claim(self, claim: str, contexts: List[str]) -> tuple:
        """Verify if claim is supported by contexts."""
        if self.method == "embedding":
            return self._verify_embedding(claim, contexts)
        elif self.method == "nli":
            return self._verify_nli(claim, contexts)
        else:  # claim_extraction
            return self._verify_claim_extraction(claim, contexts)

    def _verify_embedding(self, claim: str, contexts: List[str]) -> tuple:
        """Verify using embedding similarity."""
        if self.embedding_model is None:
            raise ValueError("Embedding model required for embedding verification")

        claim_emb = self.embedding_model.encode(claim)

        best_similarity = 0
        best_evidence = None

        for ctx in contexts:
            # Check sentence-level
            sentences = ctx.split(".")
            for sent in sentences:
                sent_emb = self.embedding_model.encode(sent)
                similarity = float(
                    np.dot(claim_emb, sent_emb)
                    / (np.linalg.norm(claim_emb) * np.linalg.norm(sent_emb) + 1e-8)
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_evidence = sent

        is_supported = best_similarity > self.threshold
        return is_supported, best_evidence

    def _verify_nli(self, claim: str, contexts: List[str]) -> tuple:
        """Verify using Natural Language Inference."""
        if self.nli_model is None:
            raise ValueError("NLI model required for NLI verification")

        # Concatenate contexts
        context_text = " ".join(contexts)

        # NLI: premise=context, hypothesis=claim
        try:
            result = self.nli_model(context_text, claim)
            # entailment = supported, contradiction = unsupported, neutral = uncertain
            label = result.get("label", "neutral")
            score = result.get("score", 0.5)

            is_supported = label == "entailment" and score > 0.7
            evidence = context_text[:200] if is_supported else None

            return is_supported, evidence
        except Exception:
            return False, None

    def _verify_claim_extraction(self, claim: str, contexts: List[str]) -> tuple:
        """Verify by extracting similar claims from context."""
        # Simplified: check for overlapping key terms
        claim_terms = set(claim.lower().split())

        best_overlap = 0
        best_evidence = None

        for ctx in contexts:
            sentences = ctx.split(".")
            for sent in sentences:
                sent_terms = set(sent.lower().split())
                overlap = len(claim_terms & sent_terms) / len(claim_terms) if claim_terms else 0

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_evidence = sent

        is_supported = best_overlap > 0.5
        return is_supported, best_evidence
