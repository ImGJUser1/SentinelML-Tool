from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/traditional/drift/mmd_detector.py
"""
Maximum Mean Discrepancy (MMD) for multivariate drift detection.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

from sentinelml.traditional.drift.base import BaseDriftDetector

logger = logging.getLogger(__name__)


class MMDDriftDetector(BaseDriftDetector):
    """
    Maximum Mean Discrepancy for multivariate drift detection.

    MMD measures the distance between mean embeddings of two distributions
    in a reproducing kernel Hilbert space (RKHS). It captures multivariate
    dependencies that univariate tests miss.

    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel type ('rbf', 'linear', 'polynomial').
    gamma : float, optional
        Kernel bandwidth. If None, uses median heuristic.
    n_permutations : int, default=1000
        Number of permutations for p-value estimation.
    chunk_size : int, default=1000
        Process data in chunks to manage memory.

    Examples
    --------
    >>> detector = MMDDriftDetector(kernel='rbf', gamma=0.5)
    >>> detector.fit(X_reference)
    >>> is_drift, p_values = detector.detect(X_test)  # Multivariate test
    """

    def __init__(
        self,
        name: str = "MMDDriftDetector",
        window_size: int = 1000,
        threshold: float = 0.05,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        n_permutations: int = 1000,
        chunk_size: int = 1000,
        verbose: bool = False,
    ):
        super().__init__(name=name, window_size=window_size, threshold=threshold, verbose=verbose)
        self.kernel = kernel
        self.gamma = gamma
        self.n_permutations = n_permutations
        self.chunk_size = chunk_size
        self.gamma_: Optional[float] = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "MMDDriftDetector":
        """Fit and compute kernel bandwidth if needed."""
        X = np.asarray(X)
        super().fit(X, y)

        # Median heuristic for bandwidth
        if self.gamma is None:
            dists = cdist(X[: min(1000, len(X))], X[: min(1000, len(X))], metric="euclidean")
            self.gamma_ = 1.0 / np.median(dists[dists > 0])
        else:
            self.gamma_ = self.gamma

        return self

    def detect(self, X: npt.ArrayLike) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """
        Compute MMD between reference and new data.

        Uses permutation test for statistical significance.

        Returns
        -------
        is_drift : ndarray
            Boolean drift flags.
        p_values : ndarray
            Statistical significance (lower = more drift).
        """
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_ref = len(self.reference_data_)
        n_new = len(X)

        # Compute MMD statistic
        mmd_stat = self._compute_mmd(self.reference_data_, X)

        # Permutation test for p-value
        combined = np.vstack([self.reference_data_, X])
        permuted_stats = []

        for _ in range(self.n_permutations):
            np.random.shuffle(combined)
            perm_ref = combined[:n_ref]
            perm_new = combined[n_ref:]
            perm_mmd = self._compute_mmd(perm_ref, perm_new)
            permuted_stats.append(perm_mmd)

        p_value = np.mean(np.array(permuted_stats) >= mmd_stat)

        # Return per-sample results (same value for batch)
        is_drift = np.array([p_value < self.threshold] * n_new)
        p_values = np.array([p_value] * n_new)

        return is_drift, p_values

    def _compute_mmd(self, X: npt.NDArray, Y: npt.NDArray) -> float:
        """
        Compute MMD^2 between two samples.

        MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        """
        K_xx = self._kernel_matrix(X, X)
        K_yy = self._kernel_matrix(Y, Y)
        K_xy = self._kernel_matrix(X, Y)

        mmd_sq = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return np.sqrt(max(mmd_sq, 0))  # Handle numerical errors

    def _kernel_matrix(self, X: npt.NDArray, Y: npt.NDArray) -> npt.NDArray:
        """Compute kernel matrix between two sets of samples."""
        if self.kernel == "rbf":
            dists = cdist(X, Y, metric="sqeuclidean")
            return np.exp(-self.gamma_ * dists)
        elif self.kernel == "linear":
            return X @ Y.T
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
