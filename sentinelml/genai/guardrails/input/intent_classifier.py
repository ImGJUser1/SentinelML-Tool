import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/genai/guardrails/input/intent_classifier.py
"""
Intent classification for off-topic detection.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseGuardrail


class IntentClassifier(BaseGuardrail):
    """
    Classify user intent and filter off-topic queries.

    Ensures inputs are relevant to the intended use case,
    preventing prompt injection via off-topic requests.

    Parameters
    ----------
    allowed_intents : list
        List of allowed intent labels.
    embedding_model : callable
        Model to encode text.
    similarity_threshold : float, default=0.7
        Minimum similarity to allowed intents.
    examples : dict, optional
        Example queries per intent for few-shot classification.

    Examples
    --------
    >>> classifier = IntentClassifier(
    ...     allowed_intents=['question_answering', 'summarization'],
    ...     embedding_model=sentence_encoder,
    ...     examples={
    ...         'question_answering': ['What is...', 'How does...'],
    ...         'summarization': ['Summarize this', 'TL;DR']
    ...     }
    ... )
    >>> result = classifier.validate("Write me a poem")  # Off-topic
    >>> print(result['is_valid'])  # False
    """

    def __init__(
        self,
        allowed_intents: List[str],
        embedding_model: Optional[Any] = None,
        name: str = "IntentClassifier",
        fail_mode: str = "filter",
        similarity_threshold: float = 0.7,
        examples: Optional[Dict[str, List[str]]] = None,
        model_name: str = "facebook/bart-large-mnli",
        verbose: bool = False,
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)
        self.allowed_intents = allowed_intents
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.examples = examples or {}
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._intent_embeddings = None

    def fit(self, X=None, y=None):
        """Compute intent embeddings."""
        if self.embedding_model is not None:
            # Encode allowed intents
            self._intent_embeddings = {}
            for intent in self.allowed_intents:
                # Use examples if available, else intent name
                texts = self.examples.get(intent, [intent.replace("_", " ")])
                embeddings = [self.embedding_model.encode(t) for t in texts]
                self._intent_embeddings[intent] = np.mean(embeddings, axis=0)
        elif self.method == "zero_shot":
            # Load zero-shot classifier
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            except ImportError:
                raise ImportError("transformers required for zero-shot classification")

        self.is_fitted_ = True
        return self

    def validate(self, content: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Classify intent and check if allowed.

        Returns
        -------
        dict with intent classification results.
        """
        if self.embedding_model is not None:
            return self._validate_embedding(content)
        else:
            return self._validate_zero_shot(content)

    def _validate_embedding(self, content: str) -> Dict[str, Any]:
        """Validate using embedding similarity."""
        content_emb = self.embedding_model.encode(content)

        # Compute similarities to all allowed intents
        similarities = {}
        for intent, intent_emb in self._intent_embeddings.items():
            sim = np.dot(content_emb, intent_emb) / (
                np.linalg.norm(content_emb) * np.linalg.norm(intent_emb) + 1e-8
            )
            similarities[intent] = float(sim)

        # Get best matching intent
        best_intent = max(similarities, key=similarities.get)
        best_score = similarities[best_intent]

        is_allowed = best_score >= self.similarity_threshold

        return {
            "is_valid": is_allowed,
            "score": best_score if is_allowed else best_score * 0.5,
            "action": "pass" if is_allowed else self.fail_mode,
            "metadata": {
                "predicted_intent": best_intent,
                "confidence": best_score,
                "all_similarities": similarities,
                "threshold": self.similarity_threshold,
            },
        }

    def _validate_zero_shot(self, content: str) -> Dict[str, Any]:
        """Validate using zero-shot classification."""
        import torch

        # Create hypothesis for each intent
        hypotheses = [f"This text is about {intent}." for intent in self.allowed_intents]

        # Tokenize premise and hypotheses
        inputs = self._tokenizer(
            [content] * len(hypotheses),
            hypotheses,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        # Get entailment scores
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        # Extract entailment probabilities
        entailment_probs = torch.softmax(logits, dim=-1)[:, 1]  # Entailment class

        # Get best intent
        best_idx = torch.argmax(entailment_probs).item()
        best_score = entailment_probs[best_idx].item()
        best_intent = self.allowed_intents[best_idx]

        is_allowed = best_score >= self.similarity_threshold

        return {
            "is_valid": is_allowed,
            "score": best_score,
            "action": "pass" if is_allowed else self.fail_mode,
            "metadata": {
                "predicted_intent": best_intent,
                "confidence": best_score,
                "all_scores": {
                    intent: entailment_probs[i].item()
                    for i, intent in enumerate(self.allowed_intents)
                },
            },
        }
