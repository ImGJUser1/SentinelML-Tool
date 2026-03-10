import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/adapters/sklearn_adapter.py
"""
Scikit-learn compatibility adapter.
"""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseSentinelComponent


class SklearnAdapter(BaseSentinelComponent):
    """
    Make SentinelML components work with scikit-learn pipelines.

    Wraps SentinelML detectors and trust models as sklearn
    transformers and meta-estimators.

    Parameters
    ----------
    sentinel_component : object
        SentinelML component to adapt.
    method : str, default='transform'
        Sklearn method to implement.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sentinelml.traditional import MMDDriftDetector

    >>> drift_detector = MMDDriftDetector()
    >>> adapter = SklearnAdapter(drift_detector, method='transform')

    >>> pipeline = Pipeline([
    ...     ('drift', adapter),
    ...     ('classifier', RandomForestClassifier())
    ... ])
    """

    def __init__(
        self,
        sentinel_component: Any,
        name: str = "SklearnAdapter",
        method: str = "transform",
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.component = sentinel_component
        self.method = method

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "SklearnAdapter":
        """Fit the wrapped component."""
        if hasattr(self.component, "fit"):
            self.component.fit(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray:
        """Transform data (for drift detectors, trust models)."""
        X = np.asarray(X)

        if self.method == "transform":
            if hasattr(self.component, "detect"):
                _, scores = self.component.detect(X)
                return scores.reshape(-1, 1)
            elif hasattr(self.component, "score"):
                scores = self.component.score(X)
                return scores.reshape(-1, 1)
            else:
                raise ValueError("Component has no transform method")

        return X

    def predict(self, X: npt.ArrayLike) -> npt.NDArray:
        """Predict (for guardrails, validators)."""
        X = np.asarray(X)

        if hasattr(self.component, "validate"):
            results = [self.component.validate(x) for x in X]
            return np.array([r["is_valid"] for r in results])

        if hasattr(self.component, "detect"):
            is_drift, _ = self.component.detect(X)
            return is_drift

        if hasattr(self.component, "score"):
            scores = self.component.score(X)
            return (scores > 0.5).astype(int)

        raise ValueError("Component has no predict method")

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray:
        """Predict probabilities."""
        X = np.asarray(X)

        if hasattr(self.component, "score"):
            scores = self.component.score(X)
            # Return [1-score, score] for binary classification format
            return np.column_stack([1 - scores, scores])

        if hasattr(self.component, "detect"):
            _, scores = self.component.detect(X)
            # Convert p-values to probabilities
            probs = 1 - scores
            return np.column_stack([1 - probs, probs])

        raise ValueError("Component has no predict_proba method")

    def score(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> float:
        """Compute score (for sklearn compatibility)."""
        if hasattr(self.component, "score"):
            return float(np.mean(self.component.score(X)))
        return 0.0

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters."""
        return {"sentinel_component": self.component, "method": self.method}

    def set_params(self, **params):
        """Set parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
