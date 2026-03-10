import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/traditional/familiarity/kernel_density.py
"""
Kernel Density Estimation for non-parametric familiarity.
"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import KernelDensity

from sentinelml.core.base import BaseTrustModel


class KernelDensityFamiliarity(BaseTrustModel):
    """
    Familiarity using Kernel Density Estimation.

    Non-parametric density estimation using Gaussian kernels.
    More flexible than distance-based methods for complex distributions.

    Parameters
    ----------
    bandwidth : float, default=1.0
        Kernel bandwidth (standard deviation).
    kernel : str, default='gaussian'
        Kernel type ('gaussian', 'tophat', 'epanechnikov').
    atol : float, default=0
        Absolute tolerance for tree-based acceleration.

    Examples
    --------
    >>> familiarity = KernelDensityFamiliarity(bandwidth=0.5)
    >>> familiarity.fit(X_train)
    >>> log_densities = familiarity.score_samples(X_test)
    """

    def __init__(
        self,
        name: str = "KernelDensityFamiliarity",
        calibration_method: str = "isotonic",
        bandwidth: float = 1.0,
        kernel: str = "gaussian",
        atol: float = 0,
        rtol: float = 0,
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.atol = atol
        self.rtol = rtol
        self.kde_ = None
        self.max_log_prob_ = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "KernelDensityFamiliarity":
        """Fit KDE to reference data."""
        X = np.asarray(X)

        self.kde_ = KernelDensity(
            bandwidth=self.bandwidth, kernel=self.kernel, atol=self.atol, rtol=self.rtol
        )
        self.kde_.fit(X)

        # Store max log probability for normalization
        self.max_log_prob_ = self.kde_.score_samples(X).max()

        self.is_fitted_ = True
        return self

    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Compute familiarity as normalized density.

        Returns
        -------
        scores : ndarray
            Familiarity in [0, 1].
        """
        self._check_is_fitted()
        X = np.asarray(X)

        log_probs = self.kde_.score_samples(X)
        # Normalize to [0, 1] using sigmoid-like transformation
        scores = np.exp(log_probs - self.max_log_prob_)
        return np.clip(scores, 0, 1)

    def score_samples(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return raw log probability densities."""
        self._check_is_fitted()
        X = np.asarray(X)
        return self.kde_.score_samples(X)

    def sample(self, n_samples: int = 1) -> npt.NDArray:
        """Generate samples from estimated density."""
        self._check_is_fitted()
        return self.kde_.sample(n_samples)
