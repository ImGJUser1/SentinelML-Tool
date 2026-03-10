from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/genai/guardrails/input/toxicity_filter.py
"""
Toxicity detection for input filtering.
"""

import re
from typing import Any, Dict, List, Optional

from sentinelml.core.base import BaseGuardrail


class ToxicityFilter(BaseGuardrail):
    """
    Filter toxic, harmful, or inappropriate content.

    Uses pattern matching, keyword lists, and optional
    transformer-based toxicity classification.

    Parameters
    ----------
    method : str, default='keyword'
        Detection method ('keyword', 'model', 'hybrid').
    threshold : float, default=0.8
        Confidence threshold for model-based detection.
    custom_keywords : list, optional
        Additional keywords to filter.
    languages : list, default=['en']
        Languages to check.

    Examples
    --------
    >>> filter = ToxicityFilter(method='hybrid', threshold=0.7)
    >>> result = filter.validate("Toxic message here...")
    >>> print(result['action'])  # 'block' if toxic
    """

    # Comprehensive toxicity patterns
    TOXIC_PATTERNS = {
        "hate_speech": [
            r"\b(hate|hating)\b.*\b(group|race|religion|gender)\b",
            r"\b(kill|die|death to)\b.*\b(all|every)\b",
        ],
        "harassment": [
            r"\b(stupid|idiot|moron|dumb)\b",
            r"\b(shut up|get lost|go away)\b",
        ],
        "profanity": [
            r"\b(fuck|shit|damn|ass|bitch)\b",
        ],
        "threats": [
            r"\b(will|gonna|going to)\b.*\b(hurt|kill|attack|destroy)\b",
            r"\b(better watch out|be careful)\b",
        ],
        "self_harm": [
            r"\b(kill myself|suicide|end it all)\b",
            r"\b(hurt myself|self.?harm)\b",
        ],
    }

    def __init__(
        self,
        name: str = "ToxicityFilter",
        fail_mode: str = "block",
        method: str = "keyword",
        threshold: float = 0.8,
        custom_keywords: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        model_name: str = "unitary/toxic-bert",
        verbose: bool = False,
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)
        self.method = method
        self.threshold = threshold
        self.custom_keywords = custom_keywords or []
        self.languages = languages or ["en"]
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

        # Compile patterns
        self._patterns = {}
        for category, patterns in self.TOXIC_PATTERNS.items():
            self._patterns[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def fit(self, X=None, y=None):
        """Load model if using ML-based detection."""
        if self.method in ("model", "hybrid"):
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            except ImportError:
                raise ImportError("transformers required for model-based toxicity detection")
        self.is_fitted_ = True
        return self

    def validate(self, content: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate content for toxicity.

        Returns
        -------
        dict with validation results.
        """
        content_lower = content.lower()
        detected_categories = []
        scores = {}

        # Keyword-based detection
        if self.method in ("keyword", "hybrid"):
            for category, patterns in self._patterns.items():
                for pattern in patterns:
                    if pattern.search(content):
                        detected_categories.append(category)
                        scores[category] = scores.get(category, 0) + 0.3
                        break

            # Custom keywords
            for keyword in self.custom_keywords:
                if keyword.lower() in content_lower:
                    detected_categories.append("custom")
                    scores["custom"] = scores.get("custom", 0) + 0.5

        # Model-based detection
        if self.method in ("model", "hybrid") and self._model is not None:
            import torch

            inputs = self._tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.sigmoid(outputs.logits).squeeze()

                # Multi-label toxicity
                labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
                for i, label in enumerate(labels):
                    score = probs[i].item()
                    scores[label] = score
                    if score > self.threshold:
                        detected_categories.append(label)

        # Aggregate score
        max_score = max(scores.values()) if scores else 0.0
        is_toxic = len(detected_categories) > 0 or max_score > self.threshold

        return {
            "is_valid": not is_toxic,
            "score": 1.0 - min(max_score, 1.0),
            "action": "pass" if not is_toxic else self.fail_mode,
            "metadata": {
                "detected_categories": list(set(detected_categories)),
                "category_scores": scores,
                "threshold": self.threshold,
                "method": self.method,
            },
        }
