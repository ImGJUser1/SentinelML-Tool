import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/core/ensemble.py
"""
Adaptive ensemble methods for combining multiple trust signals.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from sentinelml.core.base import BaseTrustModel

logger = logging.getLogger(__name__)


class AdaptiveTrustEnsemble(BaseTrustModel):
    """
    Dynamically weighted ensemble of trust models.

    Optimizes weights to maximize separation between correct and
    incorrect predictions on validation data.

    Parameters
    ----------
    validators : list of BaseTrustModel
        Trust models to ensemble.
    optimization_method : str, default='nelder-mead'
        Optimization algorithm for weight learning.
    min_weight : float, default=0.05
        Minimum weight for any validator (prevents exclusion).
    """

    def __init__(
        self,
        validators: List[BaseTrustModel],
        name: str = "AdaptiveEnsemble",
        calibration_method: str = "isotonic",
        optimization_method: str = "nelder-mead",
        min_weight: float = 0.05,
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.validators = validators
        self.optimization_method = optimization_method
        self.min_weight = min_weight
        self.weights_: Optional[npt.NDArray[np.float64]] = None
        self.validation_scores_: Optional[Dict[str, npt.NDArray]] = None

    def fit(
        self,
        X: npt.ArrayLike,
        y: Optional[npt.NDArray] = None,
        predictions: Optional[npt.NDArray] = None,
        true_labels: Optional[npt.NDArray] = None,
    ) -> "AdaptiveTrustEnsemble":
        """
        Fit ensemble and learn optimal weights.

        Parameters
        ----------
        X : array-like
            Validation features.
        y : array-like, optional
            Target values (for supervised validators).
        predictions : array-like, optional
            Model predictions on X.
        true_labels : array-like, optional
            Ground truth labels (required for weight optimization).
        """
        X = np.asarray(X)
        self._log(f"Fitting {len(self.validators)} validators")

        # Fit individual validators
        for validator in self.validators:
            try:
                validator.fit(X, y)
            except Exception as e:
                logger.warning(f"Failed to fit {validator.name}: {e}")

        # Compute validation scores
        self.validation_scores_ = {}
        for validator in self.validators:
            try:
                scores = validator.score(X)
                self.validation_scores_[validator.name] = scores
            except Exception as e:
                logger.warning(f"Failed to score with {validator.name}: {e}")
                self.validation_scores_[validator.name] = np.ones(len(X)) * 0.5

        # Optimize weights if ground truth available
        if true_labels is not None and predictions is not None:
            self._optimize_weights(true_labels, predictions)
        else:
            # Equal weights
            n = len(self.validators)
            self.weights_ = np.ones(n) / n

        self.is_fitted_ = True
        return self

    def _optimize_weights(self, true_labels: npt.NDArray, predictions: npt.NDArray):
        """Optimize weights to maximize trust separation."""
        correct = (predictions == true_labels).astype(float)

        def objective(weights: npt.NDArray) -> float:
            """Negative separation score (to minimize)."""
            weights = np.maximum(weights, self.min_weight)
            weights /= weights.sum()

            # Compute weighted ensemble score
            ensemble_scores = np.zeros(len(true_labels))
            for i, (name, scores) in enumerate(self.validation_scores_.items()):
                ensemble_scores += weights[i] * scores

            # Compute separation: high trust accuracy - low trust accuracy
            high_threshold = np.percentile(ensemble_scores, 70)
            low_threshold = np.percentile(ensemble_scores, 30)

            high_mask = ensemble_scores >= high_threshold
            low_mask = ensemble_scores <= low_threshold

            if high_mask.sum() == 0 or low_mask.sum() == 0:
                return 0.0

            high_acc = correct[high_mask].mean()
            low_acc = correct[low_mask].mean()

            separation = high_acc - low_acc
            return -separation  # Minimize negative separation

        # Initialize with equal weights
        x0 = np.ones(len(self.validators)) / len(self.validators)

        # Constrained optimization
        bounds = [(self.min_weight, 1.0) for _ in self.validators]
        constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        result = minimize(
            objective,
            x0,
            method=self.optimization_method,
            bounds=bounds,
            constraints=constraint,
            options={"maxiter": 1000},
        )

        self.weights_ = result.x / result.x.sum()
        self._log(
            f"Optimized weights: {dict(zip([v.name for v in self.validators], self.weights_))}"
        )

    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Compute ensemble trust score.

        Returns
        -------
        scores : ndarray
            Weighted combination of validator scores.
        """
        self._check_is_fitted()
        X = np.asarray(X)

        ensemble_scores = np.zeros(len(X))
        for i, validator in enumerate(self.validators):
            try:
                scores = validator.score(X)
                ensemble_scores += self.weights_[i] * scores
            except Exception as e:
                logger.warning(f"Scoring failed for {validator.name}: {e}")

        return np.clip(ensemble_scores, 0, 1)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return validator weights as importance scores."""
        if self.weights_ is None:
            raise RuntimeError("Ensemble not fitted")
        return {
            validator.name: float(weight)
            for validator, weight in zip(self.validators, self.weights_)
        }
