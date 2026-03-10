from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/traditional/drift/psi_detector.py
"""
Population Stability Index (PSI) for drift detection.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from sentinelml.traditional.drift.base import BaseDriftDetector


class PSIDetector(BaseDriftDetector):
    """
    Population Stability Index for monitoring distribution shifts.

    PSI is commonly used in credit risk modeling and is interpretable:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.25: Moderate change
    - PSI >= 0.25: Significant change

    Parameters
    ----------
    n_bins : int, default=10
        Number of bins for discretization.
    mode : str, default='equal_width'
        Binning strategy ('equal_width', 'equal_freq', 'reference').

    Examples
    --------
    >>> detector = PSIDetector(threshold=0.2)
    >>> detector.fit(X_train)
    >>> is_drift, psi_scores = detector.detect(X_test)
    """

    # PSI interpretation thresholds
    INTERPRETATION = {
        "no_change": (0, 0.1),
        "moderate": (0.1, 0.25),
        "significant": (0.25, float("inf")),
    }

    def __init__(
        self,
        name: str = "PSIDetector",
        window_size: int = 1000,
        threshold: float = 0.25,
        n_bins: int = 10,
        mode: str = "equal_width",
        verbose: bool = False,
    ):
        super().__init__(name=name, window_size=window_size, threshold=threshold, verbose=verbose)
        self.n_bins = n_bins
        self.mode = mode
        self.bins_: Optional[list] = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "PSIDetector":
        """Fit binning strategy to reference data."""
        X = np.asarray(X)
        super().fit(X, y)

        # Create bins for each feature
        self.bins_ = []
        for i in range(X.shape[1]):
            if self.mode == "equal_width":
                bins = np.linspace(X[:, i].min(), X[:, i].max(), self.n_bins + 1)
            elif self.mode == "equal_freq":
                bins = np.percentile(X[:, i], np.linspace(0, 100, self.n_bins + 1))
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            self.bins_.append(bins)

        return self

    def detect(self, X: npt.ArrayLike) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """
        Compute PSI between reference and new data.

        Returns
        -------
        is_drift : ndarray
            True if PSI exceeds threshold.
        psi_scores : ndarray
            Maximum PSI across features per sample.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = len(X)
        psi_scores = np.zeros((n_samples, X.shape[1]))

        for feature_idx in range(X.shape[1]):
            ref_dist = self._get_distribution(self.reference_data_[:, feature_idx], feature_idx)

            for sample_idx in range(n_samples):
                new_dist = self._get_distribution([X[sample_idx, feature_idx]], feature_idx)
                psi = self._compute_psi(ref_dist, new_dist)
                psi_scores[sample_idx, feature_idx] = psi

        max_psi = np.max(psi_scores, axis=1)
        is_drift = max_psi > self.threshold

        return is_drift, max_psi

    def _get_distribution(self, values: npt.ArrayLike, feature_idx: int) -> npt.NDArray:
        """Convert values to probability distribution over bins."""
        values = np.asarray(values)
        bins = self.bins_[feature_idx]

        # Handle edge cases
        if len(values) == 0:
            return np.ones(self.n_bins) / self.n_bins

        counts, _ = np.histogram(values, bins=bins)

        # Add smoothing to avoid division by zero
        counts = counts + 1e-10
        return counts / counts.sum()

    def _compute_psi(self, expected: npt.NDArray, actual: npt.NDArray) -> float:
        """Compute PSI between two distributions."""
        # Avoid log(0) by clipping
        expected = np.clip(expected, 1e-10, 1)
        actual = np.clip(actual, 1e-10, 1)

        psi = np.sum((actual - expected) * np.log(actual / expected))
        return float(psi)

    def interpret_psi(self, psi_value: float) -> str:
        """Return human-readable interpretation of PSI value."""
        for level, (low, high) in self.INTERPRETATION.items():
            if low <= psi_value < high:
                return level
        return "unknown"
