import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/traditional/trust/isolation_forest.py
"""
Isolation Forest-based trust scoring.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import IsolationForest

from sentinelml.core.base import BaseTrustModel


class IsolationForestTrust(BaseTrustModel):
    """
    Trust scoring using Isolation Forest anomaly detection.

    Isolation Forest is efficient for high-dimensional data and
    isolates anomalies instead of profiling normal data.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of base estimators.
    contamination : float, default='auto'
        Expected proportion of outliers.
    max_samples : int, default=256
        Number of samples to draw for training each base estimator.

    Examples
    --------
    >>> trust_model = IsolationForestTrust(n_estimators=200)
    >>> trust_model.fit(X_reference)
    >>> scores = trust_model.score(X_test)
    """

    def __init__(
        self,
        name: str = "IsolationForestTrust",
        calibration_method: str = "isotonic",
        n_estimators: int = 100,
        contamination: Union[str, float] = "auto",
        max_samples: Union[int, float] = 256,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.model_: Optional[IsolationForest] = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "IsolationForestTrust":
        """Fit Isolation Forest to reference data."""
        X = np.asarray(X)

        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X)
        self.is_fitted_ = True
        return self

    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Compute trust scores.

        Returns
        -------
        scores : ndarray
            Trust scores where 1 = inlier (trustworthy), 0 = outlier.
        """
        self._check_is_fitted()
        X = np.asarray(X)

        # Get anomaly scores (-1 for outliers, 1 for inliers)
        # Convert to [0, 1] trust scores
        raw_scores = self.model_.score_samples(X)  # Lower = more anomalous
        # Normalize to [0, 1]
        scores = 1 - (raw_scores.min() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-10)
        return np.clip(scores, 0, 1)
