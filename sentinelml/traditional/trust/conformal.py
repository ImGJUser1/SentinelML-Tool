import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/traditional/trust/conformal.py
"""
Conformal prediction for uncertainty quantification.
"""

from typing import List, Optional

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseTrustModel


class ConformalPredictor(BaseTrustModel):
    """
    Conformal prediction sets with guaranteed coverage.

    Provides prediction sets with statistical guarantees
    and converts set size to trust scores.

    Parameters
    ----------
    base_model : object
        Base classifier with predict_proba.
    alpha : float, default=0.1
        Significance level (1-alpha coverage).
    method : str, default='split'
        Conformal method ('split', 'cv', 'jackknife').

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> base_model = RandomForestClassifier()
    >>> conformal = ConformalPredictor(base_model, alpha=0.05)
    >>> conformal.fit(X_calib, y_calib)
    >>> sets, trust_scores = conformal.predict(X_test)
    """

    def __init__(
        self,
        base_model: Any,
        name: str = "ConformalPredictor",
        calibration_method: str = "isotonic",
        alpha: float = 0.1,
        method: str = "split",
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.base_model = base_model
        self.alpha = alpha
        self.method = method
        self.calibration_scores_ = None
        self.n_classes_ = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "ConformalPredictor":
        """
        Fit base model and compute calibration scores.

        Parameters
        ----------
        X : array-like
            Calibration features.
        y : array-like
            Calibration labels.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if y is None:
            raise ValueError("Conformal prediction requires labels")

        self.n_classes_ = len(np.unique(y))

        if self.method == "split":
            # Split conformal: train on first half, calibrate on second
            n = len(X)
            split = n // 2

            # Train base model
            self.base_model.fit(X[:split], y[:split])

            # Compute non-conformity scores on calibration set
            calib_probs = self.base_model.predict_proba(X[split:])
            calib_true = y[split:]

            # Scores = 1 - probability of true class
            self.calibration_scores_ = 1 - calib_probs[np.arange(len(calib_true)), calib_true]

        elif self.method == "cv":
            # Cross-validation conformal
            from sklearn.model_selection import cross_val_predict

            probs = cross_val_predict(self.base_model, X, y, method="predict_proba", cv=5)
            self.calibration_scores_ = 1 - probs[np.arange(len(y)), y]
            self.base_model.fit(X, y)

        self.is_fitted_ = True
        return self

    def predict(self, X: npt.ArrayLike) -> Tuple[List[List[int]], npt.NDArray[np.float64]]:
        """
        Generate prediction sets and trust scores.

        Returns
        -------
        prediction_sets : list of lists
            Sets of predicted class labels per sample.
        trust_scores : ndarray
            Trust based on set size (smaller = more confident).
        """
        self._check_is_fitted()
        X = np.asarray(X)

        # Get probabilities
        probs = self.base_model.predict_proba(X)

        # Compute quantile for threshold
        q = np.quantile(
            self.calibration_scores_,
            np.ceil((len(self.calibration_scores_) + 1) * (1 - self.alpha))
            / len(self.calibration_scores_),
        )

        prediction_sets = []
        for prob in probs:
            # Include classes where 1 - prob <= q
            set_mask = (1 - prob) <= q
            pred_set = np.where(set_mask)[0].tolist()
            prediction_sets.append(pred_set)

        # Trust = 1 / set_size (normalized)
        set_sizes = np.array([len(s) for s in prediction_sets])
        trust_scores = 1 - (set_sizes - 1) / (self.n_classes_ - 1 + 1e-8)
        trust_scores = np.clip(trust_scores, 0, 1)

        return prediction_sets, trust_scores

    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return trust scores only."""
        _, trust_scores = self.predict(X)
        return trust_scores

    def get_coverage_guarantee(self) -> float:
        """Return theoretical coverage guarantee."""
        return 1 - self.alpha
