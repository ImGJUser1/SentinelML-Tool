import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/rag/generation/citation_accuracy.py
"""
Citation accuracy verification for RAG answers.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from sentinelml.core.base import BaseSentinelComponent


class CitationAccuracy(BaseSentinelComponent):
    """
    Verify accuracy of citations in RAG-generated answers.

    Ensures that [1], [2], etc. citations actually support
    the claims they are attached to.

    Parameters
    ----------
    citation_pattern : str
        Regex to extract citations.
    verify_quotes : bool, default=True
        Verify quoted text appears in cited source.

    Examples
    --------
    >>> verifier = CitationAccuracy()
    >>> result = verifier.verify(
    ...     answer="The sky is blue [1].",
    ...     contexts=["The sky appears blue due to Rayleigh scattering."],
    ...     citations={1: "The sky appears blue..."}
    ... )
    """

    DEFAULT_CITATION_PATTERN = r"\[(\d+)\]"

    def __init__(
        self,
        name: str = "CitationAccuracy",
        citation_pattern: str = None,
        verify_quotes: bool = True,
        similarity_threshold: float = 0.8,
        embedding_model: Optional[Any] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.citation_pattern = citation_pattern or self.DEFAULT_CITATION_PATTERN
        self.verify_quotes = verify_quotes
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def verify(
        self, answer: str, contexts: List[str], citations: Optional[Dict[int, str]] = None
    ) -> Dict[str, Any]:
        """
        Verify citation accuracy.

        Parameters
        ----------
        answer : str
            Generated answer with citations.
        contexts : list
            Retrieved contexts (indexed by citation number).
        citations : dict, optional
            Mapping of citation number to context text.

        Returns
        -------
        dict with verification results.
        """
        # Extract citations from answer
        citation_matches = re.findall(self.citation_pattern, answer)
        cited_indices = [int(c) for c in citation_matches]

        if not cited_indices:
            return {
                "accuracy_score": 1.0,  # No citations to check
                "citations_found": 0,
                "accurate_citations": [],
                "inaccurate_citations": [],
                "uncited_claims": [],
            }

        # Map citations to contexts
        if citations is None:
            citations = {i + 1: ctx for i, ctx in enumerate(contexts)}

        accurate = []
        inaccurate = []

        # Check each citation
        for cite_num in set(cited_indices):
            # Find the claim associated with this citation
            claim = self._extract_claim_for_citation(answer, cite_num)

            # Get cited source
            source = citations.get(cite_num)
            if source is None:
                inaccurate.append(
                    {
                        "citation_number": cite_num,
                        "claim": claim,
                        "issue": "Citation index out of range",
                    }
                )
                continue

            # Verify claim against source
            is_accurate, evidence = self._verify_claim_against_source(claim, source)

            if is_accurate:
                accurate.append(
                    {"citation_number": cite_num, "claim": claim, "source_preview": source[:200]}
                )
            else:
                inaccurate.append(
                    {
                        "citation_number": cite_num,
                        "claim": claim,
                        "source_preview": source[:200],
                        "issue": "Claim not supported by cited source",
                    }
                )

        # Calculate accuracy
        total = len(set(cited_indices))
        accuracy = len(accurate) / total if total > 0 else 1.0

        # Check for uncited claims
        uncited = self._find_uncited_claims(answer, cited_indices)

        return {
            "accuracy_score": accuracy,
            "citations_found": total,
            "accurate_citations": accurate,
            "inaccurate_citations": inaccurate,
            "uncited_claims": uncited,
            "is_accurate": accuracy >= 0.8 and len(inaccurate) == 0,
        }

    def _extract_claim_for_citation(self, answer: str, cite_num: int) -> str:
        """Extract the claim associated with a citation."""
        # Find citation in text
        pattern = rf"([^.]*?\[{cite_num}\][^.]*\.)"
        match = re.search(pattern, answer)

        if match:
            return match.group(1).strip()

        # Fallback: return sentence containing citation
        sentences = answer.split(".")
        for sent in sentences:
            if f"[{cite_num}]" in sent:
                return sent.strip()

        return ""

    def _verify_claim_against_source(self, claim: str, source: str) -> Tuple[bool, str]:
        """Verify if claim is supported by source."""
        if self.embedding_model is not None:
            claim_emb = self.embedding_model.encode(claim)
            source_emb = self.embedding_model.encode(source)

            similarity = float(
                np.dot(claim_emb, source_emb)
                / (np.linalg.norm(claim_emb) * np.linalg.norm(source_emb) + 1e-8)
            )

            return similarity > self.similarity_threshold, source[:200]

        # Keyword fallback
        claim_words = set(claim.lower().split())
        source_words = set(source.lower().split())

        overlap = len(claim_words & source_words) / len(claim_words) if claim_words else 0
        return overlap > 0.5, source[:200]

    def _find_uncited_claims(self, answer: str, cited_indices: List[int]) -> List[str]:
        """Find factual claims without citations."""
        sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]

        uncited = []
        for sent in sentences:
            # Check if sentence has any citation
            has_citation = any(f"[{c}]" in sent for c in cited_indices)

            # Check if it looks like a factual claim
            is_factual = any(w in sent.lower() for w in ["is", "are", "was", "were", "has", "have"])

            if is_factual and not has_citation:
                uncited.append(sent[:100] + "..." if len(sent) > 100 else sent)

        return uncited[:5]  # Limit to 5
