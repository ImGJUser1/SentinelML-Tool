import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/traditional/familiarity/kdtree.py
"""
KDTree-based familiarity scoring.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import KDTree

from sentinelml.core.base import BaseTrustModel


class KDTreeFamiliarity(BaseTrustModel):
    """
    Local density estimation using KDTree.

    Familiarity is based on distance to k-th nearest neighbor
    in reference data, scaled by median distance.

    Parameters
    ----------
    k : int, default=5
        Number of neighbors for density estimation.
    metric : str, default='euclidean'
        Distance metric.

    Examples
    --------
    >>> familiarity = KDTreeFamiliarity(k=10)
    >>> familiarity.fit(X_train)
    >>> scores = familiarity.score(X_test)
    """

    def __init__(
        self,
        name: str = "KDTreeFamiliarity",
        calibration_method: str = "isotonic",
        k: int = 5,
        metric: str = "euclidean",
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.k = k
        self.metric = metric
        self.tree_: Optional[KDTree] = None
        self.scale_: Optional[float] = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "KDTreeFamiliarity":
        """Build KDTree on reference data."""
        X = np.asarray(X)
        self.tree_ = KDTree(X, metric=self.metric)

        # Compute scale as median distance to k-th neighbor
        dists, _ = self.tree_.query(X, k=self.k + 1)
        self.scale_ = np.median(dists[:, -1]) + 1e-8

        self.is_fitted_ = True
        return self

    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Compute familiarity scores.

        Returns
        -------
        scores : ndarray
            Familiarity in [0, 1], higher = more similar to reference.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        dists, _ = self.tree_.query(X, k=1)
        scores = np.exp(-dists.squeeze() / self.scale_)
        return scores
