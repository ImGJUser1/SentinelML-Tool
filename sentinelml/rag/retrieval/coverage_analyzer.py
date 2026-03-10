from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/rag/retrieval/coverage_analyzer.py
"""
Analyze information coverage in retrieved documents.
"""

from typing import Any, Dict, List, Optional, Set

import numpy as np

from sentinelml.core.base import BaseSentinelComponent


class CoverageAnalyzer(BaseSentinelComponent):
    """
    Assess how well retrieved documents cover query aspects.

    Identifies missing information and coverage gaps
    in the retrieval results.

    Parameters
    ----------
    embedding_model : callable
        Model for semantic comparison.
    aspect_extractor : callable, optional
        Function to extract aspects from query.

    Examples
    --------
    >>> analyzer = CoverageAnalyzer(embedding_model=encoder)
    >>> coverage = analyzer.analyze(query, retrieved_docs)
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        aspect_extractor: Optional[Any] = None,
        name: str = "CoverageAnalyzer",
        threshold: float = 0.7,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.embedding_model = embedding_model
        self.aspect_extractor = aspect_extractor
        self.threshold = threshold

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def analyze(self, query: str, documents: List[str]) -> Dict[str, Any]:
        """
        Analyze coverage of query aspects in documents.

        Returns
        -------
        dict with coverage analysis.
        """
        # Extract aspects from query
        aspects = self._extract_aspects(query)

        if not aspects:
            return {"coverage_score": 1.0, "aspects_covered": [], "aspects_missed": [], "gaps": []}

        # Check coverage for each aspect
        covered_aspects = []
        missed_aspects = []

        for aspect in aspects:
            is_covered = self._check_aspect_coverage(aspect, documents)
            if is_covered:
                covered_aspects.append(aspect)
            else:
                missed_aspects.append(aspect)

        coverage_score = len(covered_aspects) / len(aspects)

        return {
            "coverage_score": coverage_score,
            "aspects_covered": covered_aspects,
            "aspects_missed": missed_aspects,
            "n_aspects": len(aspects),
            "n_covered": len(covered_aspects),
            "gaps": self._identify_gaps(missed_aspects, documents),
        }

    def _extract_aspects(self, query: str) -> List[str]:
        """Extract key aspects/topics from query."""
        if self.aspect_extractor is not None:
            return self.aspect_extractor(query)

        # Simple extraction: noun phrases and key terms
        import re

        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)

        # Extract key terms (capitalized or after "about", "regarding")
        terms = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query)

        # Extract after prepositions
        prepositional = re.findall(
            r"\b(?:about|regarding|on|for)\s+([a-z]+(?:\s+[a-z]+){0,3})", query, re.IGNORECASE
        )

        aspects = list(set(quoted + terms + prepositional))
        return aspects[:10]  # Limit to top 10

    def _check_aspect_coverage(self, aspect: str, documents: List[str]) -> bool:
        """Check if aspect is covered in any document."""
        if self.embedding_model is not None:
            aspect_emb = self.embedding_model.encode(aspect)

            for doc in documents:
                # Check sentence-level coverage
                sentences = doc.split(".")
                for sent in sentences:
                    sent_emb = self.embedding_model.encode(sent)
                    similarity = float(
                        np.dot(aspect_emb, sent_emb)
                        / (np.linalg.norm(aspect_emb) * np.linalg.norm(sent_emb) + 1e-8)
                    )
                    if similarity > self.threshold:
                        return True
            return False
        else:
            # Keyword matching fallback
            aspect_words = set(aspect.lower().split())
            for doc in documents:
                doc_words = set(doc.lower().split())
                overlap = len(aspect_words & doc_words) / len(aspect_words) if aspect_words else 0
                if overlap > 0.5:
                    return True
            return False

    def _identify_gaps(self, missed_aspects: List[str], documents: List[str]) -> List[Dict]:
        """Identify specific information gaps."""
        gaps = []
        for aspect in missed_aspects:
            # Find most similar document (to show near-miss)
            best_doc_idx = None
            best_similarity = 0

            if self.embedding_model is not None:
                aspect_emb = self.embedding_model.encode(aspect)
                for i, doc in enumerate(documents):
                    doc_emb = self.embedding_model.encode(doc[:500])  # First 500 chars
                    sim = float(
                        np.dot(aspect_emb, doc_emb)
                        / (np.linalg.norm(aspect_emb) * np.linalg.norm(doc_emb) + 1e-8)
                    )
                    if sim > best_similarity:
                        best_similarity = sim
                        best_doc_idx = i

            gaps.append(
                {
                    "aspect": aspect,
                    "closest_document": best_doc_idx,
                    "similarity_to_closest": best_similarity,
                    "suggestion": f"Add information about {aspect}",
                }
            )

        return gaps
