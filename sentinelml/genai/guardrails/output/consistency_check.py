from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/genai/guardrails/output/consistency_check.py
"""
Self-consistency and logical coherence checking.
"""

import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseGuardrail


class ConsistencyCheck(BaseGuardrail):
    """
    Check output for internal consistency and logical coherence.

    Detects contradictions, tautologies, and logical fallacies
    within the generated text.

    Parameters
    ----------
    check_types : list, default=['contradiction', 'tautology']
        Types of checks to perform.
    n_paraphrases : int, default=3
        Number of paraphrases for self-consistency.
    similarity_threshold : float, default=0.8
        Threshold for semantic equivalence.

    Examples
    --------
    >>> checker = ConsistencyCheck(check_types=['contradiction', 'temporal'])
    >>> result = checker.validate("It is Monday. Today is Tuesday.")
    >>> print(result['metadata']['inconsistencies'])
    """

    def __init__(
        self,
        name: str = "ConsistencyCheck",
        fail_mode: str = "flag",
        check_types: Optional[List[str]] = None,
        n_paraphrases: int = 3,
        similarity_threshold: float = 0.8,
        paraphrase_model: Optional[Callable] = None,
        embedding_model: Optional[Callable] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)
        self.check_types = check_types or ["contradiction", "tautology"]
        self.n_paraphrases = n_paraphrases
        self.similarity_threshold = similarity_threshold
        self.paraphrase_model = paraphrase_model
        self.embedding_model = embedding_model

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def validate(self, content: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Check content for consistency issues.

        Returns
        -------
        dict with consistency analysis.
        """
        inconsistencies = []
        scores = []

        # Extract statements/sentences
        statements = self._extract_statements(content)

        # Check for contradictions
        if "contradiction" in self.check_types:
            contra_score, contra_issues = self._check_contradictions(statements)
            scores.append(contra_score)
            inconsistencies.extend(contra_issues)

        # Check for tautologies
        if "tautology" in self.check_types:
            taut_score, taut_issues = self._check_tautologies(statements)
            scores.append(taut_score)
            inconsistencies.extend(taut_issues)

        # Check temporal consistency
        if "temporal" in self.check_types:
            temp_score, temp_issues = self._check_temporal_consistency(statements)
            scores.append(temp_score)
            inconsistencies.extend(temp_issues)

        # Self-consistency via paraphrasing
        if "paraphrase" in self.check_types and self.paraphrase_model is not None:
            para_score, para_issues = self._check_paraphrase_consistency(content)
            scores.append(para_score)
            inconsistencies.extend(para_issues)

        # Aggregate score
        avg_score = np.mean(scores) if scores else 1.0
        is_consistent = len(inconsistencies) == 0

        return {
            "is_valid": is_consistent,
            "score": avg_score,
            "action": "pass" if is_consistent else self.fail_mode,
            "metadata": {
                "inconsistencies": inconsistencies,
                "n_statements": len(statements),
                "check_types": self.check_types,
                "per_check_scores": dict(zip(self.check_types, scores)) if scores else {},
            },
        }

    def _extract_statements(self, text: str) -> List[str]:
        """Extract individual statements from text."""
        # Simple sentence splitting
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _check_contradictions(self, statements: List[str]) -> Tuple[float, List[Dict]]:
        """Check for contradictory statements."""
        issues = []

        # Simple negation detection
        negation_words = ["not", "n't", "never", "no", "none", "nothing", "nobody"]

        for i, stmt1 in enumerate(statements):
            for j, stmt2 in enumerate(statements[i + 1 :], i + 1):
                # Check if one contains negation and other doesn't
                stmt1_neg = any(w in stmt1.lower() for w in negation_words)
                stmt2_neg = any(w in stmt2.lower() for w in negation_words)

                if stmt1_neg != stmt2_neg:
                    # High similarity with opposite negation = contradiction
                    similarity = self._text_similarity(stmt1, stmt2)
                    if similarity > 0.8:
                        issues.append(
                            {
                                "type": "contradiction",
                                "statement_1": stmt1,
                                "statement_2": stmt2,
                                "similarity": similarity,
                                "severity": "high" if similarity > 0.9 else "medium",
                            }
                        )

        score = 1.0 - (len(issues) * 0.2)
        return max(score, 0.0), issues

    def _check_tautologies(self, statements: List[str]) -> Tuple[float, List[Dict]]:
        """Check for tautological statements."""
        issues = []
        tautology_patterns = [
            r"\b(A|a)\s+is\s+\1\b",  # "A is A"
            r"\b(it is what it is)\b",
            r"\b(always|never)\s+.*\b(always|never)\b",
        ]

        for stmt in statements:
            for pattern in tautology_patterns:
                if re.search(pattern, stmt):
                    issues.append({"type": "tautology", "statement": stmt, "pattern": pattern})

        score = 1.0 - (len(issues) * 0.1)
        return max(score, 0.0), issues

    def _check_temporal_consistency(self, statements: List[str]) -> Tuple[float, List[Dict]]:
        """Check for temporal contradictions."""
        issues = []

        # Simple temporal expression extraction
        temporal_markers = {
            "past": ["yesterday", "last week", "ago", "previously", "before"],
            "present": ["today", "now", "currently", "at present"],
            "future": ["tomorrow", "next week", "soon", "later", "will"],
        }

        statement_tenses = []
        for stmt in statements:
            tenses = []
            for tense, markers in temporal_markers.items():
                if any(m in stmt.lower() for m in markers):
                    tenses.append(tense)
            statement_tenses.append(tenses)

        # Check for conflicting tenses in related statements
        for i, (stmt1, tenses1) in enumerate(zip(statements, statement_tenses)):
            for j, (stmt2, tenses2) in enumerate(
                zip(statements[i + 1 :], statement_tenses[i + 1 :]), i + 1
            ):
                if tenses1 and tenses2:
                    # Check if tenses conflict (simplified)
                    if ("past" in tenses1 and "future" in tenses2) or (
                        "future" in tenses1 and "past" in tenses2
                    ):
                        similarity = self._text_similarity(stmt1, stmt2)
                        if similarity > 0.5:
                            issues.append(
                                {
                                    "type": "temporal_inconsistency",
                                    "statement_1": stmt1,
                                    "statement_2": stmt2,
                                    "tense_1": tenses1,
                                    "tense_2": tenses2,
                                }
                            )

        score = 1.0 - (len(issues) * 0.15)
        return max(score, 0.0), issues

    def _check_paraphrase_consistency(self, content: str) -> Tuple[float, List[Dict]]:
        """Check consistency across paraphrases."""
        if self.paraphrase_model is None:
            return 1.0, []

        # Generate paraphrases
        paraphrases = []
        for _ in range(self.n_paraphrases):
            try:
                para = self.paraphrase_model(content)
                paraphrases.append(para)
            except Exception:
                continue

        if len(paraphrases) < 2:
            return 1.0, []

        # Check semantic equivalence
        similarities = []
        for i, p1 in enumerate(paraphrases):
            for p2 in paraphrases[i + 1 :]:
                sim = self._text_similarity(p1, p2)
                similarities.append(sim)

        avg_similarity = np.mean(similarities)
        is_consistent = avg_similarity > self.similarity_threshold

        issues = []
        if not is_consistent:
            issues.append(
                {
                    "type": "paraphrase_inconsistency",
                    "avg_similarity": avg_similarity,
                    "paraphrases": paraphrases,
                }
            )

        return avg_similarity, issues

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between texts."""
        if self.embedding_model is not None:
            emb1 = self.embedding_model.encode(text1)
            emb2 = self.embedding_model.encode(text2)
            return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
        else:
            # Fallback to Jaccard similarity
            set1 = set(text1.lower().split())
            set2 = set(text2.lower().split())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
