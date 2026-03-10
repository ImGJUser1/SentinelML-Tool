from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/rag/advanced/claim_verification.py
"""
Decompose-then-verify for complex claim verification.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from sentinelml.core.base import BaseSentinelComponent


class ClaimVerifier(BaseSentinelComponent):
    """
    Verify complex claims by decomposition.

    Breaks down complex claims into atomic facts,
    verifies each independently, and aggregates results.

    Parameters
    ----------
    llm_client : callable
        For claim decomposition.
    base_verifier : object
        Verifier for atomic facts.

    Examples
    --------
    >>> verifier = ClaimVerifier(llm_client=openai, base_verifier=faithfulness_checker)
    >>> result = verifier.verify(
    ...     "The Eiffel Tower was built in 1889 and is 330m tall.",
    ...     contexts
    ... )
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        base_verifier: Optional[Any] = None,
        name: str = "ClaimVerifier",
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.llm_client = llm_client
        self.base_verifier = base_verifier

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def verify(self, claim: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Verify a complex claim by decomposition.

        Returns
        -------
        dict with verification results.
        """
        # Decompose claim into atomic facts
        atomic_facts = self._decompose_claim(claim)

        if not atomic_facts:
            # Try to verify as single claim
            if self.base_verifier:
                return self.base_verifier.check(claim, contexts)
            return {"verified": False, "reason": "Could not decompose"}

        # Verify each atomic fact
        verified_facts = []
        unverified_facts = []

        for fact in atomic_facts:
            if self.base_verifier:
                result = self.base_verifier.check(fact, contexts)
                is_verified = result.get("faithfulness_score", 0) > 0.7
            else:
                is_verified = self._simple_verify(fact, contexts)

            if is_verified:
                verified_facts.append(fact)
            else:
                unverified_facts.append(fact)

        # Aggregate results
        verification_rate = len(verified_facts) / len(atomic_facts)

        return {
            "verified": verification_rate > 0.8 and len(unverified_facts) == 0,
            "verification_rate": verification_rate,
            "atomic_facts": atomic_facts,
            "verified_facts": verified_facts,
            "unverified_facts": unverified_facts,
            "original_claim": claim,
        }

    def _decompose_claim(self, claim: str) -> List[str]:
        """Decompose complex claim into atomic facts."""
        if self.llm_client is None:
            # Rule-based decomposition
            return self._rule_based_decomposition(claim)

        # LLM-based decomposition
        prompt = f"""
        Break down the following claim into simple, atomic facts that can be verified independently.
        Each fact should be a single, verifiable statement.

        Claim: {claim}

        Format your response as:
        Fact 1: [first atomic fact]
        Fact 2: [second atomic fact]
        ...
        """

        try:
            response = self.llm_client(prompt)
            return self._parse_decomposition(response)
        except Exception:
            return self._rule_based_decomposition(claim)

    def _rule_based_decomposition(self, claim: str) -> List[str]:
        """Simple rule-based claim decomposition."""
        # Split by conjunctions
        separators = [" and ", " but ", " while ", " whereas ", "; ", ". "]

        facts = [claim]
        for sep in separators:
            new_facts = []
            for f in facts:
                parts = f.split(sep)
                new_facts.extend([p.strip() for p in parts if len(p.strip()) > 10])
            facts = new_facts

        # Filter out questions and fragments
        facts = [f for f in facts if not f.endswith("?") and len(f.split()) >= 3]

        return facts[:10]  # Limit to 10 facts

    def _parse_decomposition(self, response: str) -> List[str]:
        """Parse LLM decomposition response."""
        facts = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.lower().startswith("fact ") and ":" in line:
                fact = line.split(":", 1)[1].strip()
                if fact:
                    facts.append(fact)
        return facts

    def _simple_verify(self, fact: str, contexts: List[str]) -> bool:
        """Simple verification without base verifier."""
        fact_words = set(fact.lower().split())

        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            overlap = len(fact_words & ctx_words) / len(fact_words) if fact_words else 0
            if overlap > 0.6:
                return True

        return False
