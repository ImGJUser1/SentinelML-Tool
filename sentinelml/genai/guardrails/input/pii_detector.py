from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/genai/guardrails/input/pii_detector.py
"""
Personally Identifiable Information (PII) detection.
"""

import re
from typing import Any, Dict, List, Optional, Set

from sentinelml.core.base import BaseGuardrail


class PIIDetector(BaseGuardrail):
    """
    Detect and filter PII in inputs.

    Supports regex patterns for common PII types and
    optional NER-based detection with transformers.

    Parameters
    ----------
    entities : list, default=['email', 'phone', 'ssn', 'credit_card']
        PII types to detect.
    method : str, default='regex'
        Detection method ('regex', 'ner', 'hybrid').
        redact : bool, default=True
        Redact detected PII in output.
    custom_patterns : dict, optional
        Custom regex patterns {entity: pattern}.

    Examples
    --------
    >>> detector = PIIDetector(entities=['email', 'phone'])
    >>> result = detector.validate("Contact me at user@email.com")
    >>> print(result['metadata']['detected_entities'])
    """

    # Standard PII patterns
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "ssn": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-.\s]?){3}\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        "api_key": r'\b(?:api[_-]?key|apikey)[\s]*[=:][\s]*["\']?[a-zA-Z0-9]{16,}["\']?',
        "password": r'\b(?:password|pwd|passwd)[\s]*[=:][\s]*["\']?[^\s"\']+["\']?',
        "name": r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)?\s*[A-Z][a-z]+\s+[A-Z][a-z]+\b",
    }

    def __init__(
        self,
        name: str = "PIIDetector",
        fail_mode: str = "sanitize",
        entities: Optional[List[str]] = None,
        method: str = "regex",
        redact: bool = True,
        custom_patterns: Optional[Dict[str, str]] = None,
        model_name: str = "dslim/bert-base-NER",
        verbose: bool = False,
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)
        self.entities = entities or ["email", "phone", "ssn", "credit_card"]
        self.method = method
        self.redact = redact
        self.custom_patterns = custom_patterns or {}
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

        # Compile patterns
        self._patterns = {}
        all_patterns = {**self.PII_PATTERNS, **self.custom_patterns}
        for entity in self.entities:
            if entity in all_patterns:
                self._patterns[entity] = re.compile(all_patterns[entity])

    def fit(self, X=None, y=None):
        """Load NER model if needed."""
        if self.method in ("ner", "hybrid"):
            try:
                from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForTokenClassification.from_pretrained(self.model_name)
                self._ner_pipeline = pipeline(
                    "ner",
                    model=self._model,
                    tokenizer=self._tokenizer,
                    aggregation_strategy="simple",
                )
            except ImportError:
                raise ImportError("transformers required for NER-based detection")
        self.is_fitted_ = True
        return self

    def validate(self, content: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Detect PII in content.

        Returns
        -------
        dict with detection results and sanitized content.
        """
        detected_entities = []
        entity_spans = []  # (start, end, entity_type)

        # Regex-based detection
        if self.method in ("regex", "hybrid"):
            for entity_type, pattern in self._patterns.items():
                for match in pattern.finditer(content):
                    detected_entities.append(
                        {
                            "type": entity_type,
                            "value": match.group(),
                            "start": match.start(),
                            "end": match.end(),
                            "method": "regex",
                        }
                    )
                    entity_spans.append((match.start(), match.end(), entity_type))

        # NER-based detection
        if self.method in ("ner", "hybrid") and self._ner_pipeline is not None:
            ner_results = self._ner_pipeline(content)
            for entity in ner_results:
                if entity["entity_group"] in ["PER", "ORG", "LOC"]:
                    detected_entities.append(
                        {
                            "type": entity["entity_group"],
                            "value": entity["word"],
                            "start": entity["start"],
                            "end": entity["end"],
                            "method": "ner",
                            "score": entity["score"],
                        }
                    )
                    entity_spans.append((entity["start"], entity["end"], entity["entity_group"]))

        # Check if any entities detected
        has_pii = len(detected_entities) > 0

        # Sanitize if requested
        sanitized_content = content
        if has_pii and self.redact:
            # Sort spans by start position (reverse for safe replacement)
            sorted_spans = sorted(entity_spans, key=lambda x: x[0], reverse=True)
            for start, end, entity_type in sorted_spans:
                sanitized_content = (
                    sanitized_content[:start]
                    + f"[REDACTED_{entity_type}]"
                    + sanitized_content[end:]
                )

        # Determine action
        if has_pii:
            if self.fail_mode == "sanitize":
                action = "sanitize"
            elif self.fail_mode == "block":
                action = "block"
            else:
                action = "flag"
        else:
            action = "pass"

        return {
            "is_valid": not has_pii or self.fail_mode == "sanitize",
            "score": 1.0 if not has_pii else 0.5,
            "action": action,
            "sanitized_content": sanitized_content if has_pii and self.redact else None,
            "metadata": {
                "detected_entities": detected_entities,
                "entity_count": len(detected_entities),
                "entity_types": list(set(e["type"] for e in detected_entities)),
                "redacted": has_pii and self.redact,
            },
        }

    def redact_content(self, content: str) -> str:
        """Redact PII from content."""
        result = self.validate(content)
        return result.get("sanitized_content") or content
