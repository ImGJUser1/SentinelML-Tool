import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/deep_learning/uncertainty/temperature_scaling.py
"""
Temperature scaling for model calibration.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from sentinelml.core.base import BaseTrustModel


class TemperatureScaling(BaseTrustModel):
    """
    Post-hoc calibration using temperature scaling.

    Learns a single temperature parameter to soften
    softmax probabilities, improving calibration.

    Parameters
    ----------
    temperature : float, optional
        Initial temperature. If None, learned from data.
    max_iter : int, default=1000
        Optimization iterations.

    Examples
    --------
    >>> calibrator = TemperatureScaling()
    >>> calibrator.fit(X_val, y_val, model=model)
    >>> calibrated_probs = calibrator.predict_proba(X_test)
    """

    def __init__(
        self,
        name: str = "TemperatureScaling",
        calibration_method: str = "isotonic",
        temperature: Optional[float] = None,
        max_iter: int = 1000,
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.temperature = temperature
        self.max_iter = max_iter
        self.temperature_ = temperature
        self.model_ = None

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None, model: Optional[Callable] = None
    ) -> "TemperatureScaling":
        """
        Fit temperature on validation set.

        Parameters
        ----------
        X : array-like
            Validation features.
        y : array-like
            Validation labels.
        model : callable
            Model with predict_proba or predict.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if model is None and self.model_ is None:
            raise ValueError("Model required for temperature scaling")

        if model is not None:
            self.model_ = model

        # Get uncalibrated probabilities
        if hasattr(self.model_, "predict_proba"):
            logits = self.model_.predict_proba(X)
            # Convert to logit-like (inverse softmax)
            logits = np.log(np.clip(logits, 1e-10, 1))
        else:
            # Assume logits directly
            logits = self.model_.predict(X)

        # Optimize temperature
        if self.temperature is None:

            def nll_loss(T):
                scaled = logits / T
                exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
                probs = exp_scaled / exp_scaled.sum(axis=-1, keepdims=True)
                log_probs = np.log(probs[np.arange(len(y)), y] + 1e-10)
                return -log_probs.mean()

            result = minimize_scalar(
                nll_loss, bounds=(0.1, 10.0), method="bounded", options={"maxiter": self.max_iter}
            )
            self.temperature_ = result.x
        else:
            self.temperature_ = self.temperature

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Get calibrated probabilities."""
        self._check_is_fitted()
        X = np.asarray(X)

        if hasattr(self.model_, "predict_proba"):
            logits = self.model_.predict_proba(X)
            logits = np.log(np.clip(logits, 1e-10, 1))
        else:
            logits = self.model_.predict(X)

        # Apply temperature
        scaled = logits / self.temperature_
        exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        probs = exp_scaled / exp_scaled.sum(axis=-1, keepdims=True)
        return probs

    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Trust based on maximum calibrated probability.

        Returns
        -------
        scores : ndarray
            Trust scores (max probability after calibration).
        """
        probs = self.predict_proba(X)
        max_probs = probs.max(axis=-1)
        return max_probs

    def get_temperature(self) -> float:
        """Return learned temperature."""
        self._check_is_fitted()
        return self.temperature_
