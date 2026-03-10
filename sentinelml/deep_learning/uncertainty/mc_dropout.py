import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/deep_learning/uncertainty/mc_dropout.py
"""
Monte Carlo Dropout for uncertainty estimation.
"""

from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseTrustModel


class MCDropoutUncertainty(BaseTrustModel):
    """
    Bayesian uncertainty estimation using MC Dropout.

    Performs multiple forward passes with dropout enabled to
    estimate epistemic uncertainty.

    Parameters
    ----------
    model : callable
        PyTorch/TensorFlow model with dropout layers.
    n_samples : int, default=50
        Number of MC samples.
    dropout_rate : float, optional
        Override model dropout rate.

    Examples
    --------
    >>> model = create_dropout_model()
    >>> uncertainty_est = MCDropoutUncertainty(model, n_samples=100)
    >>> uncertainty_est.fit(X_calib)
    >>> scores = uncertainty_est.score(X_test)
    """

    def __init__(
        self,
        model: Callable,
        name: str = "MCDropoutUncertainty",
        n_samples: int = 50,
        dropout_rate: Optional[float] = None,
        calibration_method: str = "isotonic",
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        self._is_torch = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "MCDropoutUncertainty":
        """Calibration data for uncertainty scaling."""
        X = np.asarray(X)
        # Store calibration statistics
        self._calibration_mean = np.zeros(X.shape[1])
        self._calibration_std = np.ones(X.shape[1])
        self.is_fitted_ = True
        return self

    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Compute uncertainty-based trust scores.

        Returns
        -------
        scores : ndarray
            Trust scores (1 - normalized uncertainty).
        """
        self._check_is_fitted()
        X = np.asarray(X)

        # Detect framework
        if self._is_torch is None:
            self._is_torch = self._check_is_torch()

        if self._is_torch:
            return self._score_torch(X)
        else:
            return self._score_tensorflow(X)

    def _check_is_torch(self) -> bool:
        """Check if model is PyTorch."""
        try:
            import torch

            return isinstance(self.model, torch.nn.Module)
        except ImportError:
            return False

    def _score_torch(self, X: npt.NDArray) -> npt.NDArray[np.float64]:
        """PyTorch implementation."""
        import torch

        self.model.train()  # Enable dropout
        X_tensor = torch.tensor(X, dtype=torch.float32)

        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                out = self.model(X_tensor)
                predictions.append(out.cpu().numpy())

        predictions = np.array(predictions)  # (n_samples, n_batch, n_classes)

        # Compute uncertainty as predictive entropy
        mean_pred = predictions.mean(axis=0)
        uncertainty = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=-1)

        # Convert to trust score
        trust = 1 - (uncertainty / np.log(predictions.shape[-1]))  # Normalize by max entropy
        return np.clip(trust, 0, 1)

    def _score_tensorflow(self, X: npt.NDArray) -> npt.NDArray[np.float64]:
        """TensorFlow implementation."""
        import tensorflow as tf

        predictions = []
        for _ in range(self.n_samples):
            out = self.model(X, training=True)  # Enable dropout
            predictions.append(out.numpy())

        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        uncertainty = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=-1)
        trust = 1 - (uncertainty / np.log(predictions.shape[-1]))
        return np.clip(trust, 0, 1)
