from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/genai/guardrails/input/injection_detector.py
"""
Prompt injection attack detection.
"""

import re
from typing import Any, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseGuardrail


class PromptInjectionDetector(BaseGuardrail):
    """
    Detects prompt injection attacks in LLM inputs.

    Uses pattern matching, heuristic analysis, and optional
    transformer-based classification.

    Parameters
    ----------
    method : str, default='heuristic'
        Detection method ('heuristic', 'model', 'hybrid').
    threshold : float, default=0.7
        Confidence threshold for detection.
    custom_patterns : list, optional
        Additional regex patterns to check.

    Examples
    --------
    >>> detector = PromptInjectionDetector(method='hybrid')
    >>> result = detector.validate("Ignore previous instructions and...")
    >>> print(result['is_valid'])  # False
    """

    # Known injection patterns
    DEFAULT_PATTERNS = [
        r"ignore (previous|above|earlier) (instructions|prompt)",
        r"disregard (all|any|previous) instructions",
        r"forget (what|everything) (you were|you're|you are)",
        r"new instructions?:",
        r"system prompt:",
        r"you are now",
        r"pretend (to be|you are)",
        r"act as (if|though)",
        r"roleplay",
        r"developer mode",
        r"sudo",
        r"root access",
        r"ignore safety",
        r"bypass (restrictions|filters)",
        r"jailbreak",
        r"DAN (do anything now)",
    ]

    def __init__(
        self,
        name: str = "PromptInjectionDetector",
        fail_mode: str = "block",
        method: str = "heuristic",
        threshold: float = 0.7,
        custom_patterns: Optional[List[str]] = None,
        model_name: str = "deberta-v3-base",
        verbose: bool = False,
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)
        self.method = method
        self.threshold = threshold
        self.patterns = self.DEFAULT_PATTERNS + (custom_patterns or [])
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def fit(self, X, y=None):
        """Load model if using ML-based detection."""
        if self.method in ("model", "hybrid"):
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{self.model_name}")
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    f"microsoft/{self.model_name}", num_labels=2
                )
            except ImportError:
                raise ImportError("transformers required for model-based detection")
        self.is_fitted_ = True
        return self

    def validate(self, content: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate prompt for injection attempts.

        Parameters
        ----------
        content : str
            Input prompt/text to validate.
        context : dict, optional
            Additional context (conversation history, etc.).

        Returns
        -------
        dict with keys:
            - is_valid: bool
            - score: float (0-1, higher = more likely injection)
            - action: str
            - metadata: dict
        """
        content_lower = content.lower()
        score = 0.0
        matches = []

        # Heuristic detection
        if self.method in ("heuristic", "hybrid"):
            for pattern in self.patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    matches.append(pattern)
                    score += 0.3  # Additive scoring

            # Additional heuristics
            if content.count('"') > 10 or content.count("'") > 10:
                score += 0.1  # Excessive quoting

            if len(content) > 2000:
                score += 0.1  # Unusually long

            # Check for delimiter confusion
            delimiters = ["```", '"""', "||", "###", "/*"]
            for delim in delimiters:
                if content.count(delim) >= 2:
                    score += 0.2

        # Model-based detection
        if self.method in ("model", "hybrid") and self._model is not None:
            import torch

            inputs = self._tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                model_score = probs[0][1].item()  # Injection class probability
                score = max(score, model_score)

        score = min(score, 1.0)
        is_valid = score < self.threshold

        return {
            "is_valid": is_valid,
            "score": 1.0 - score,  # Convert to trust score
            "action": "pass" if is_valid else self.fail_mode,
            "metadata": {
                "injection_probability": score,
                "matched_patterns": matches,
                "method": self.method,
                "threshold": self.threshold,
            },
        }
