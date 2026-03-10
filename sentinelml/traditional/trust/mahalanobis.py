import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/traditional/trust/mahalanobis.py
"""
Mahalanobis distance-based trust scoring.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.linalg import pinvh

from sentinelml.core.base import BaseTrustModel


class MahalanobisTrust(BaseTrustModel):
    """
    Trust scoring based on Mahalanobis distance from reference distribution.

    Uses robust covariance estimation with shrinkage for high-dimensional data.

    Parameters
    ----------
    robust : bool, default=True
        Use Ledoit-Wolf shrinkage for covariance estimation.
    shrinkage : float, optional
        Shrinkage parameter (0=MLE, 1=diagonal). Auto if None.

    Examples
    --------
    >>> trust_model = MahalanobisTrust(robust=True)
    >>> trust_model.fit(X_reference)
    >>> trust_scores = trust_model.score(X_test)
    """

    def __init__(
        self,
        name: str = "MahalanobisTrust",
        calibration_method: str = "isotonic",
        robust: bool = True,
        shrinkage: Optional[float] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.robust = robust
        self.shrinkage = shrinkage
        self.mean_: Optional[npt.NDArray] = None
        self.cov_inv_: Optional[npt.NDArray] = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "MahalanobisTrust":
        """
        Fit Gaussian distribution to reference data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Reference data.
        """
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)

        if self.robust:
            # Ledoit-Wolf shrinkage
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf()
            lw.fit(X)
            cov = lw.covariance_
        else:
            cov = np.cov(X.T)

        # Add small regularization and invert
        cov += np.eye(cov.shape[0]) * 1e-6
        self.cov_inv_ = pinvh(cov)  # Pseudo-inverse for stability

        self.is_fitted_ = True
        return self

    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Compute trust scores based on Mahalanobis distance.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Trust scores in [0, 1], higher = more trustworthy.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Compute Mahalanobis distance
        diff = X - self.mean_
        dists = np.sqrt(np.sum(diff @ self.cov_inv_ * diff, axis=1))

        # Convert to trust score (exponential decay)
        scores = np.exp(-dists / np.sqrt(X.shape[1]))
        return np.clip(scores, 0, 1)

    def mahalanobis_distance(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return raw Mahalanobis distances."""
        self._check_is_fitted()
        X = np.asarray(X)
        diff = X - self.mean_
        return np.sqrt(np.sum(diff @ self.cov_inv_ * diff, axis=1))
