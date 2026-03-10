import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/genai/alignment/bias_detector.py
"""
Demographic bias detection in LLM outputs.
"""

import re
from typing import Any, Dict, List, Optional, Set

from sentinelml.core.base import BaseGuardrail


class BiasDetector(BaseGuardrail):
    """
    Detect demographic biases in generated content.

    Identifies stereotypes, unequal representations,
    and potentially harmful associations.

    Parameters
    ----------
    protected_attributes : list, default=['gender', 'race', 'age']
        Attributes to monitor for bias.
    method : str, default='keyword'
        Detection method ('keyword', 'embedding', 'classifier').

    Examples
    --------
    >>> detector = BiasDetector(protected_attributes=['gender', 'race'])
    >>> result = detector.validate("All doctors are men and nurses are women.")
    >>> print(result['metadata']['bias_detected'])
    """

    BIAS_PATTERNS = {
        "gender": {
            "stereotypes": [
                r"\b(men are|women are)\s+(better|worse|more|less)\s+\w+",
                r"\b(all|most)\s+(men|women)\s+(are|have)",
                r"\b(naturally|inherently)\s+(male|female|masculine|feminine)",
            ],
            "occupations": {
                "doctor": ["he", "his", "man"],
                "nurse": ["she", "her", "woman"],
                "engineer": ["he", "his", "man"],
                "teacher": ["she", "her", "woman"],
            },
        },
        "race": {
            "stereotypes": [
                r"\b(certain|some)\s+(races|ethnicities)\s+(are|have)",
                r"\b(naturally|inherently)\s+(violent|lazy|smart|hardworking)",
                r"\b(those|these)\s+people\s+(are|always)",
            ],
        },
        "age": {
            "stereotypes": [
                r"\b(old|young)\s+people\s+(are|can\'t|don\'t)",
                r"\b(over|under)\s+\d+\s+(is|are)\s+(too)",
            ],
        },
        "religion": {
            "stereotypes": [
                r"\b(all|most)\s+(muslims|christians|jews|hindus)",
                r"\b(religious|atheist)\s+people\s+(are)",
            ],
        },
    }

    def __init__(
        self,
        name: str = "BiasDetector",
        fail_mode: str = "flag",
        protected_attributes: Optional[List[str]] = None,
        method: str = "keyword",
        threshold: float = 0.7,
        embedding_model: Optional[Any] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)
        self.protected_attributes = protected_attributes or ["gender", "race", "age"]
        self.method = method
        self.threshold = threshold
        self.embedding_model = embedding_model

    def fit(self, X=None, y=None):
        """Compile patterns."""
        self._patterns = {}
        for attr in self.protected_attributes:
            if attr in self.BIAS_PATTERNS:
                patterns = self.BIAS_PATTERNS[attr].get("stereotypes", [])
                self._patterns[attr] = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.is_fitted_ = True
        return self

    def validate(self, content: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Detect bias in content.

        Returns
        -------
        dict with bias detection results.
        """
        detected_biases = []
        scores = {}

        content_lower = content.lower()

        # Keyword-based detection
        if self.method in ("keyword", "hybrid"):
            for attr in self.protected_attributes:
                if attr not in self._patterns:
                    continue

                attr_violations = []
                for pattern in self._patterns[attr]:
                    matches = pattern.findall(content)
                    if matches:
                        attr_violations.extend(matches)

                if attr_violations:
                    detected_biases.append(
                        {
                            "attribute": attr,
                            "type": "stereotype",
                            "matches": attr_violations,
                            "severity": "medium" if len(attr_violations) < 3 else "high",
                        }
                    )
                    scores[attr] = min(len(attr_violations) * 0.3, 1.0)

        # Check for unequal representation
        representation_issues = self._check_representation(content)
        if representation_issues:
            detected_biases.extend(representation_issues)

        # Embedding-based similarity to known biased statements
        if self.method in ("embedding", "hybrid") and self.embedding_model is not None:
            embedding_score = self._embedding_bias_check(content)
            scores["embedding"] = embedding_score
            if embedding_score > self.threshold:
                detected_biases.append(
                    {
                        "attribute": "general",
                        "type": "embedding_similarity",
                        "score": embedding_score,
                    }
                )

        # Calculate overall score
        if scores:
            avg_score = 1 - np.mean(list(scores.values()))
        else:
            avg_score = 1.0

        has_bias = len(detected_biases) > 0

        return {
            "is_valid": not has_bias,
            "score": max(0.0, avg_score),
            "action": "pass" if not has_bias else self.fail_mode,
            "metadata": {
                "detected_biases": detected_biases,
                "affected_attributes": list(set(b["attribute"] for b in detected_biases)),
                "scores": scores,
                "method": self.method,
            },
        }

    def _check_representation(self, content: str) -> List[Dict]:
        """Check for unequal representation in examples."""
        issues = []

        # Simple heuristic: count pronouns
        male_pronouns = len(re.findall(r"\b(he|his|him|himself)\b", content, re.IGNORECASE))
        female_pronouns = len(re.findall(r"\b(she|her|hers|herself)\b", content, re.IGNORECASE))

        total = male_pronouns + female_pronouns
        if total > 5:  # Only check if substantial content
            ratio = max(male_pronouns, female_pronouns) / total
            if ratio > 0.8:  # Highly skewed
                issues.append(
                    {
                        "attribute": "gender",
                        "type": "representation",
                        "male_count": male_pronouns,
                        "female_count": female_pronouns,
                        "ratio": ratio,
                        "severity": "medium",
                    }
                )

        return issues

    def _embedding_bias_check(self, content: str) -> float:
        """Check similarity to known biased statements."""
        # This would compare to a database of biased statements
        # Simplified: return low score
        return 0.0
