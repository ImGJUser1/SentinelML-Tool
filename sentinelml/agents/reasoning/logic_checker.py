from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/agents/reasoning/logic_checker.py
"""
Logical consistency checking for agent reasoning.
"""

import re
from typing import Any, Dict, List, Optional

from sentinelml.core.base import BaseGuardrail


class LogicChecker(BaseGuardrail):
    """
    Check logical validity of agent reasoning steps.

    Detects logical fallacies, contradictions in reasoning,
    and invalid inference patterns.

    Parameters
    ----------
    check_fallacies : list, optional
        Specific fallacy types to check.

    Examples
    --------
    >>> checker = LogicChecker()
    >>> result = checker.validate("All A are B. C is A. Therefore C is B.")
    """

    FALLACY_PATTERNS = {
        "circular": [
            r"\b(because|since)\b.*\b(therefore|thus|so)\b.*\1",  # X because X
        ],
        "false_dichotomy": [
            r"\b(either|or)\b.*\b(otherwise|or else)\b",
            r"\bonly two (options|choices|possibilities)\b",
        ],
        "hasty_generalization": [
            r"\b(all|every|always|never)\b.*\b(some|few|one|once)\b",
        ],
        "appeal_to_authority": [
            r"\b(experts say|studies show|research proves)\b.*\b(without|no)\b.*\b(evidence|source|citation)\b",
        ],
        "slippery_slope": [
            r"\b(if|once)\b.*\b(then|will lead to|result in)\b.*\b(and then|which leads to|eventually)\b",
        ],
    }

    def __init__(
        self,
        name: str = "LogicChecker",
        fail_mode: str = "flag",
        check_fallacies: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)
        self.check_fallacies = check_fallacies or list(self.FALLACY_PATTERNS.keys())

    def fit(self, X=None, y=None):
        """Compile patterns."""
        self._patterns = {}
        for fallacy in self.check_fallacies:
            if fallacy in self.FALLACY_PATTERNS:
                import re

                self._patterns[fallacy] = [
                    re.compile(p, re.IGNORECASE) for p in self.FALLACY_PATTERNS[fallacy]
                ]
        self.is_fitted_ = True
        return self

    def validate(self, reasoning: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Check reasoning for logical issues.

        Parameters
        ----------
        reasoning : str
            Reasoning text to validate.

        Returns
        -------
        dict with logic check results.
        """
        detected_fallacies = []

        for fallacy, patterns in self._patterns.items():
            for pattern in patterns:
                matches = pattern.findall(reasoning)
                if matches:
                    detected_fallacies.append(
                        {
                            "type": fallacy,
                            "pattern": pattern.pattern,
                            "matches": matches[:3],  # Limit matches
                        }
                    )
                    break  # One match per fallacy type is enough

        # Check for explicit contradictions
        contradictions = self._find_contradictions(reasoning)

        is_valid = len(detected_fallacies) == 0 and len(contradictions) == 0

        return {
            "is_valid": is_valid,
            "score": 1.0 - (len(detected_fallacies) * 0.2) - (len(contradictions) * 0.3),
            "action": "pass" if is_valid else self.fail_mode,
            "metadata": {
                "fallacies_detected": detected_fallacies,
                "contradictions": contradictions,
                "n_issues": len(detected_fallacies) + len(contradictions),
            },
        }

    def _find_contradictions(self, reasoning: str) -> List[Dict]:
        """Find explicit contradictions in reasoning."""
        contradictions = []

        # Split into statements
        statements = [s.strip() for s in re.split(r"[.!?;]", reasoning) if len(s.strip()) > 10]

        # Look for "A but not A" patterns
        for i, stmt1 in enumerate(statements):
            # Check for negation flip
            negation_words = ["not", "n't", "never", "no", "false", "incorrect"]
            has_neg1 = any(w in stmt1.lower() for w in negation_words)

            for stmt2 in statements[i + 1 :]:
                has_neg2 = any(w in stmt2.lower() for w in negation_words)

                # Check if they're similar except for negation
                if has_neg1 != has_neg2:
                    # Remove negation words and compare
                    clean1 = " ".join(w for w in stmt1.lower().split() if w not in negation_words)
                    clean2 = " ".join(w for w in stmt2.lower().split() if w not in negation_words)

                    if clean1 == clean2 and len(clean1) > 20:
                        contradictions.append(
                            {
                                "statement_1": stmt1,
                                "statement_2": stmt2,
                                "type": "direct_contradiction",
                            }
                        )

        return contradictions[:5]  # Limit to 5
