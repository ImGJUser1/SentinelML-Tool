import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/traditional/drift/ks_univariate.py
"""
Kolmogorov-Smirnov univariate drift detection.
"""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.stats import ks_2samp

from sentinelml.traditional.drift.base import BaseDriftDetector


class KSDriftDetector(BaseDriftDetector):
    """
    Univariate Kolmogorov-Smirnov test for drift detection.

    Tests each feature independently and aggregates results.
    Suitable for low-dimensional tabular data.

    Parameters
    ----------
    correction : str, default='bonferroni'
        Multiple testing correction ('bonferroni', 'fdr', 'none').

    Examples
    --------
    >>> detector = KSDriftDetector(threshold=0.01)
    >>> detector.fit(X_reference)
    >>> is_drift, p_values = detector.detect(X_new)
    """

    def __init__(
        self,
        name: str = "KSDriftDetector",
        window_size: int = 1000,
        threshold: float = 0.05,
        correction: str = "bonferroni",
        verbose: bool = False,
    ):
        super().__init__(name=name, window_size=window_size, threshold=threshold, verbose=verbose)
        self.correction = correction

    def detect(self, X: npt.ArrayLike) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """
        Perform KS test on each feature.

        Returns
        -------
        is_drift : ndarray of shape (n_samples,)
            True if drift detected in any feature.
        p_values : ndarray of shape (n_samples,)
            Minimum p-value across features.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_features = X.shape[1]
        n_samples = len(X)

        p_values_per_feature = np.zeros((n_samples, n_features))

        for feature_idx in range(n_features):
            ref_values = self.reference_data_[:, feature_idx]

            for sample_idx in range(n_samples):
                # Compare reference against each new sample
                stat, p_value = ks_2samp(ref_values, [X[sample_idx, feature_idx]])
                p_values_per_feature[sample_idx, feature_idx] = p_value

        # Multiple testing correction
        if self.correction == "bonferroni":
            corrected_threshold = self.threshold / n_features
        else:
            corrected_threshold = self.threshold

        # Aggregate: drift if any feature shows drift
        min_p_values = np.min(p_values_per_feature, axis=1)
        is_drift = min_p_values < corrected_threshold

        return is_drift, min_p_values
