import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/rag/advanced/contradiction_detect.py
"""
Cross-document contradiction detection.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sentinelml.core.base import BaseSentinelComponent


class ContradictionDetector(BaseSentinelComponent):
    """
    Detect contradictions between retrieved documents.

    Identifies when sources disagree, which may indicate
    conflicting information or outdated sources.

    Parameters
    ----------
    embedding_model : callable
        For semantic comparison.
    nli_model : callable, optional
        Natural Language Inference model.

    Examples
    --------
    >>> detector = ContradictionDetector(embedding_model=encoder, nli_model=nli)
    >>> contradictions = detector.detect(documents)
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        nli_model: Optional[Any] = None,
        name: str = "ContradictionDetector",
        threshold: float = 0.8,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.embedding_model = embedding_model
        self.nli_model = nli_model
        self.threshold = threshold

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def detect(self, documents: List[str]) -> Dict[str, Any]:
        """
        Detect contradictions between documents.

        Returns
        -------
        dict with contradiction analysis.
        """
        if len(documents) < 2:
            return {"has_contradictions": False, "contradictions": [], "agreement_score": 1.0}

        # Extract key statements from each document
        statements_per_doc = [self._extract_statements(doc) for doc in documents]

        contradictions = []

        # Compare statements across documents
        for i, stmt_list_i in enumerate(statements_per_doc):
            for j, stmt_list_j in enumerate(statements_per_doc[i + 1 :], i + 1):
                for stmt_i in stmt_list_i:
                    for stmt_j in stmt_list_j:
                        is_contra, confidence = self._check_contradiction(stmt_i, stmt_j)

                        if is_contra:
                            contradictions.append(
                                {
                                    "doc_1_idx": i,
                                    "doc_2_idx": j,
                                    "statement_1": stmt_i,
                                    "statement_2": stmt_j,
                                    "confidence": confidence,
                                }
                            )

        # Calculate agreement score
        total_comparisons = sum(len(s) for s in statements_per_doc)
        agreement = 1 - (len(contradictions) / max(total_comparisons * 0.1, 1))

        return {
            "has_contradictions": len(contradictions) > 0,
            "n_contradictions": len(contradictions),
            "contradictions": contradictions,
            "agreement_score": max(0, agreement),
            "n_documents": len(documents),
        }

    def _extract_statements(self, document: str) -> List[str]:
        """Extract factual statements from document."""
        sentences = [s.strip() for s in document.split(".") if len(s.strip()) > 15]

        # Filter for factual statements
        factual = []
        indicators = ["is", "are", "was", "were", "has", "have", "does", "did"]

        for sent in sentences:
            if any(ind in sent.lower().split() for ind in indicators):
                # Skip questions and conditionals
                if not sent.endswith("?") and "if " not in sent.lower()[:15]:
                    factual.append(sent)

        return factual[:20]  # Limit to 20 statements

    def _check_contradiction(self, stmt1: str, stmt2: str) -> Tuple[bool, float]:
        """Check if two statements contradict."""
        # Quick check: high similarity = likely same statement, not contradiction
        if self.embedding_model:
            emb1 = self.embedding_model.encode(stmt1)
            emb2 = self.embedding_model.encode(stmt2)

            similarity = float(
                np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            )

            # Very similar = not contradiction (same info)
            if similarity > 0.9:
                return False, 0.0

            # Use NLI if available
            if self.nli_model:
                result = self.nli_model(stmt1, stmt2)
                label = result.get("label", "neutral")

                if label == "contradiction":
                    return True, result.get("score", 0.8)
                return False, 0.0

            # Heuristic: moderate similarity + negation indicators
            negation_words = ["not", "n't", "never", "no", "none", "without"]
            has_neg1 = any(w in stmt1.lower() for w in negation_words)
            has_neg2 = any(w in stmt2.lower() for w in negation_words)

            # One has negation, other doesn't, and similar topics = possible contradiction
            if has_neg1 != has_neg2 and similarity > 0.6:
                return True, similarity

        # Keyword-based fallback
        return self._keyword_contradiction_check(stmt1, stmt2)

    def _keyword_contradiction_check(self, stmt1: str, stmt2: str) -> Tuple[bool, float]:
        """Simple keyword-based contradiction detection."""
        # Check for opposing numbers
        numbers1 = re.findall(r"\b(\d+(?:\.\d+)?)\b", stmt1)
        numbers2 = re.findall(r"\b(\d+(?:\.\d+)?)\b", stmt2)

        if numbers1 and numbers2:
            # Different numbers on same topic might be contradiction
            # (simplified check)
            pass

        # Check for negation flip
        negation_words = ["not", "n't", "never", "no"]
        stmt1_neg = any(w in stmt1.lower() for w in negation_words)
        stmt2_neg = any(w in stmt2.lower() for w in negation_words)

        if stmt1_neg != stmt2_neg:
            # High word overlap with opposite negation = likely contradiction
            words1 = set(stmt1.lower().split()) - set(negation_words)
            words2 = set(stmt2.lower().split()) - set(negation_words)

            overlap = len(words1 & words2) / len(words1 | words2) if (words1 | words2) else 0

            if overlap > 0.7:
                return True, overlap

        return False, 0.0
