import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/genai/alignment/toxicity_scorer.py
"""
Google Perspective API integration for toxicity scoring.
"""

from typing import Any, Dict, Optional

import requests

from sentinelml.core.base import BaseTrustModel


class PerspectiveScorer(BaseTrustModel):
    """
    Toxicity scoring using Google's Perspective API.

    Provides scores for multiple toxicity attributes:
    - TOXICITY
    - SEVERE_TOXICITY
    - IDENTITY_ATTACK
    - INSULT
    - PROFANITY
    - THREAT

    Parameters
    ----------
    api_key : str
        Perspective API key.
    attributes : list, default=['TOXICITY']
        Which attributes to request.

    Examples
    --------
    >>> scorer = PerspectiveScorer(api_key='your_key')
    >>> scores = scorer.score("Text to analyze")
    """

    API_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

    AVAILABLE_ATTRIBUTES = [
        "TOXICITY",
        "SEVERE_TOXICITY",
        "IDENTITY_ATTACK",
        "INSULT",
        "PROFANITY",
        "THREAT",
        "SEXUALLY_EXPLICIT",
        "FLIRTATION",
        "ATTACK_ON_AUTHOR",
        "ATTACK_ON_COMMENTER",
        "INCOHERENT",
        "INFLAMMATORY",
        "OBSCENE",
        "SPAM",
        "UNSUBSTANTIAL",
    ]

    def __init__(
        self,
        api_key: str,
        name: str = "PerspectiveScorer",
        calibration_method: str = "isotonic",
        attributes: Optional[list] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.api_key = api_key
        self.attributes = attributes or ["TOXICITY"]
        self._validate_attributes()

    def _validate_attributes(self):
        """Validate requested attributes."""
        invalid = [a for a in self.attributes if a not in self.AVAILABLE_ATTRIBUTES]
        if invalid:
            raise ValueError(f"Invalid attributes: {invalid}")

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def score(self, X: str) -> float:
        """
        Get toxicity score from Perspective API.

        Returns
        -------
        score : float
            1 - toxicity (trust score).
        """
        if isinstance(X, (list, np.ndarray)):
            return np.array([self._score_single(x) for x in X])
        return self._score_single(X)

    def _score_single(self, text: str) -> float:
        """Query Perspective API."""
        data = {
            "comment": {"text": text},
            "languages": ["en"],
            "requestedAttributes": {attr: {} for attr in self.attributes},
        }

        try:
            response = requests.post(f"{self.API_URL}?key={self.api_key}", json=data, timeout=10)
            response.raise_for_status()
            result = response.json()

            # Extract scores
            scores = {}
            for attr in self.attributes:
                score = result["attributeScores"][attr]["summaryScore"]["value"]
                scores[attr] = score

            # Use maximum toxicity as overall score
            max_toxicity = max(scores.values())
            trust_score = 1 - max_toxicity

            return float(trust_score)

        except Exception as e:
            if self.verbose:
                print(f"Perspective API error: {e}")
            return 0.5  # Neutral on error

    def get_all_scores(self, text: str) -> Dict[str, float]:
        """Get all attribute scores."""
        data = {
            "comment": {"text": text},
            "languages": ["en"],
            "requestedAttributes": {attr: {} for attr in self.attributes},
        }

        response = requests.post(f"{self.API_URL}?key={self.api_key}", json=data, timeout=10)
        response.raise_for_status()
        result = response.json()

        return {
            attr: result["attributeScores"][attr]["summaryScore"]["value"]
            for attr in self.attributes
        }
