import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/deep_learning/uncertainty/evidential.py
"""
Evidential deep learning for uncertainty quantification.
"""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseTrustModel


class EvidentialNetwork(BaseTrustModel):
    """
    Evidential deep learning with explicit uncertainty modeling.

    Models parameters of Dirichlet distribution over class probabilities,
    providing explicit aleatoric and epistemic uncertainty.

    Parameters
    ----------
    model : callable
        Neural network outputting Dirichlet parameters.
    n_classes : int
        Number of classes.
    loss_type : str, default='edl'
        Loss function type ('edl', 'bayesian').

    Examples
    --------
    >>> model = create_evidential_model(n_classes=10)
    >>> evidential = EvidentialNetwork(model, n_classes=10)
    >>> evidential.fit(X_train, y_train)
    >>> trust, uncertainty = evidential.score_with_uncertainty(X_test)
    """

    def __init__(
        self,
        model: Callable,
        n_classes: int,
        name: str = "EvidentialNetwork",
        calibration_method: str = "isotonic",
        loss_type: str = "edl",
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.model = model
        self.n_classes = n_classes
        self.loss_type = loss_type
        self._is_torch = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "EvidentialNetwork":
        """Fit evidential model."""
        X = np.asarray(X)
        y = np.asarray(y) if y is not None else None

        if y is None:
            raise ValueError("Evidential learning requires labels")

        # Model should already be trained with evidential loss
        # This fit is mainly for compatibility
        self.is_fitted_ = True
        return self

    def _get_dirichlet_params(self, X: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """Get alpha (concentration) parameters from model."""
        if self._is_torch is None:
            self._is_torch = self._check_is_torch()

        if self._is_torch:
            import torch

            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                alpha = self.model(X_tensor).cpu().numpy()
            return alpha, None
        else:
            alpha = self.model.predict(X)
            return alpha, None

    def _check_is_torch(self) -> bool:
        try:
            import torch.nn as nn

            return isinstance(self.model, nn.Module)
        except ImportError:
            return False

    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Compute trust from evidential prediction.

        Uses expected probability (mean of Dirichlet).

        Returns
        -------
        scores : ndarray
            Trust scores (expected max probability).
        """
        alpha, _ = self._get_dirichlet_params(np.asarray(X))

        # Expected probability
        alpha0 = alpha.sum(axis=-1, keepdims=True)
        expected_probs = alpha / alpha0

        # Trust = max expected probability
        trust = expected_probs.max(axis=-1)
        return trust

    def score_with_uncertainty(
        self, X: npt.ArrayLike
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Return trust with uncertainty decomposition.

        Returns
        -------
        trust : ndarray
            Expected max probability.
        aleatoric : ndarray
            Data uncertainty (irreducible).
        epistemic : ndarray
            Model uncertainty (reducible with more data).
        """
        alpha, _ = self._get_dirichlet_params(np.asarray(X))

        alpha0 = alpha.sum(axis=-1, keepdims=True)
        probs = alpha / alpha0

        # Expected probability
        trust = probs.max(axis=-1)

        # Uncertainty measures
        # Aleatoric: expected entropy of categorical
        digamma_alpha0 = self._digamma(alpha0)
        digamma_alpha = self._digamma(alpha)

        aleatoric = -np.sum(probs * (digamma_alpha - digamma_alpha0), axis=-1)

        # Epistemic: variance of probabilities
        epistemic = np.sum(probs * (1 - probs) / (alpha0 + 1), axis=-1)

        return trust, aleatoric, epistemic

    def _digamma(self, x: npt.NDArray) -> npt.NDArray:
        """Compute digamma function."""
        from scipy.special import digamma

        return digamma(x)

    def get_dirichlet_parameters(self, X: npt.ArrayLike) -> npt.NDArray:
        """Return raw Dirichlet concentration parameters."""
        alpha, _ = self._get_dirichlet_params(np.asarray(X))
        return alpha
