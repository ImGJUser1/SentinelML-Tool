import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/genai/guardrails/output/citation_verifier.py
"""
Verify citations and source attributions in outputs.
"""

import re
from typing import Any, Dict, List, Optional, Set

from sentinelml.core.base import BaseGuardrail


class CitationVerifier(BaseGuardrail):
    """
    Verify that citations in outputs are accurate and supported.

    Checks for hallucinated citations, incorrect attributions,
    and unsupported claims in generated text.

    Parameters
    ----------
    knowledge_base : list of str
        Source documents that can be cited.
    citation_pattern : str, optional
        Regex pattern to extract citations.
    verify_quotes : bool, default=True
        Verify quoted text appears in sources.

    Examples
    --------
    >>> verifier = CitationVerifier(knowledge_base=documents)
    >>> result = verifier.validate(
    ...     "According to [Source A], the sky is blue.",
    ...     context={'sources': {'Source A': doc_a}}
    ... )
    """

    DEFAULT_CITATION_PATTERNS = [
        r"\[([^\]]+)\]",  # [Source Name]
        r"\(([^)]+)\s+\d{4}\)",  # (Author 2024)
        r"\"([^\"]+)\"\s*\[(\d+)\]",  # "Quote" [1]
    ]

    def __init__(
        self,
        knowledge_base: Optional[List[str]] = None,
        name: str = "CitationVerifier",
        fail_mode: str = "flag",
        citation_patterns: Optional[List[str]] = None,
        verify_quotes: bool = True,
        similarity_threshold: float = 0.85,
        embedding_model: Optional[Any] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)
        self.knowledge_base = knowledge_base or []
        self.citation_patterns = citation_patterns or self.DEFAULT_CITATION_PATTERNS
        self.verify_quotes = verify_quotes
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model

    def fit(self, X=None, y=None):
        """Index knowledge base if needed."""
        if self.embedding_model is not None and self.knowledge_base:
            # Pre-compute embeddings for efficient similarity search
            self._kb_embeddings = [self.embedding_model.encode(doc) for doc in self.knowledge_base]
        self.is_fitted_ = True
        return self

    def validate(self, content: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Verify citations in content.

        Returns
        -------
        dict with citation verification results.
        """
        sources = context.get("sources", {}) if context else {}
        citations = self._extract_citations(content)

        verified_citations = []
        unverified_citations = []
        issues = []

        for citation in citations:
            citation_text = citation["text"]
            citation_type = citation["type"]

            # Check if citation exists in provided sources
            if citation_type == "bracket":
                source_name = citation_text
                if source_name in sources:
                    verified_citations.append(
                        {"citation": citation_text, "source": source_name, "verified": True}
                    )
                else:
                    unverified_citations.append(
                        {"citation": citation_text, "issue": "Source not found in context"}
                    )
                    issues.append(f"Unverified source: {source_name}")

            elif citation_type == "quote":
                # Verify quoted text appears in sources
                quote = citation.get("quote", "")
                source_id = citation_text

                if self.verify_quotes and quote:
                    found = self._verify_quote(quote, sources.get(source_id, ""))
                    if found:
                        verified_citations.append(
                            {"quote": quote, "source": source_id, "verified": True}
                        )
                    else:
                        unverified_citations.append(
                            {
                                "quote": quote,
                                "source": source_id,
                                "issue": "Quote not found in source",
                            }
                        )
                        issues.append(f"Unverified quote from {source_id}")

        # Check for claims without citations
        uncited_claims = self._detect_uncited_claims(content, citations)

        # Calculate score
        total_citations = len(verified_citations) + len(unverified_citations)
        if total_citations == 0:
            citation_score = 1.0  # No citations to verify
        else:
            citation_score = len(verified_citations) / total_citations

        # Penalize for uncited claims
        uncited_penalty = len(uncited_claims) * 0.1
        final_score = max(0.0, citation_score - uncited_penalty)

        is_valid = len(unverified_citations) == 0 and len(uncited_claims) <= 2

        return {
            "is_valid": is_valid,
            "score": final_score,
            "action": "pass" if is_valid else self.fail_mode,
            "metadata": {
                "verified_citations": verified_citations,
                "unverified_citations": unverified_citations,
                "uncited_claims": uncited_claims,
                "total_citations": total_citations,
                "verification_rate": citation_score,
            },
        }

    def _extract_citations(self, text: str) -> List[Dict]:
        """Extract citations from text."""
        citations = []

        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) == 1:
                    citations.append(
                        {"text": match.group(1), "type": "bracket", "full_match": match.group(0)}
                    )
                elif len(match.groups()) == 2:
                    citations.append(
                        {
                            "quote": match.group(1),
                            "text": match.group(2),
                            "type": "quote",
                            "full_match": match.group(0),
                        }
                    )

        return citations

    def _verify_quote(self, quote: str, source: str) -> bool:
        """Verify that quote appears in source."""
        if not source:
            return False

        # Exact match
        if quote in source:
            return True

        # Fuzzy match with embedding similarity
        if self.embedding_model is not None:
            quote_emb = self.embedding_model.encode(quote)
            source_emb = self.embedding_model.encode(source)
            similarity = float(
                np.dot(quote_emb, source_emb)
                / (np.linalg.norm(quote_emb) * np.linalg.norm(source_emb) + 1e-8)
            )
            return similarity > self.similarity_threshold

        # Partial match
        quote_words = set(quote.lower().split())
        source_words = set(source.lower().split())
        overlap = len(quote_words & source_words) / len(quote_words) if quote_words else 0
        return overlap > 0.8

    def _detect_uncited_claims(self, text: str, citations: List[Dict]) -> List[str]:
        """Detect factual claims without citations."""
        # Simple heuristic: sentences with numbers or strong assertions
        claim_patterns = [
            r"\b(study|research|paper|report)\s+(shows|found|demonstrates)",
            r"\b(according to|as stated in)\b",
            r"\d+%\s+(of|of the)",
            r"\b(proves|confirms|establishes)\b",
        ]

        sentences = re.split(r"[.!?]+", text)
        uncited = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            # Check if sentence matches claim patterns
            is_claim = any(
                re.search(pattern, sentence, re.IGNORECASE) for pattern in claim_patterns
            )

            # Check if sentence has citation
            has_citation = any(cit["full_match"] in sentence for cit in citations)

            if is_claim and not has_citation:
                uncited.append(sentence[:100] + "..." if len(sentence) > 100 else sentence)

        return uncited[:5]  # Limit to top 5
