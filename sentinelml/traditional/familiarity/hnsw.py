import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/traditional/familiarity/hnsw.py
"""
HNSW (Hierarchical Navigable Small World) for approximate
nearest neighbor search at scale.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseTrustModel


class HNSWFamiliarity(BaseTrustModel):
    """
    Familiarity using HNSW for million-scale datasets.

    HNSW provides O(log N) query time with high recall,
    suitable for large reference datasets where KDTree fails.

    Parameters
    ----------
    M : int, default=16
        Max connections per element.
    ef_construction : int, default=200
        Size of dynamic candidate list.
    ef_search : int, default=50
        Size of search candidate list.
    k : int, default=5
        Number of neighbors for density estimation.
    metric : str, default='euclidean'
        Distance metric.

    Examples
    --------
    >>> familiarity = HNSWFamiliarity(M=32, ef_construction=400)
    >>> familiarity.fit(X_large)  # Millions of samples
    >>> scores = familiarity.score(X_test)
    """

    def __init__(
        self,
        name: str = "HNSWFamiliarity",
        calibration_method: str = "isotonic",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        k: int = 5,
        metric: str = "euclidean",
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.k = k
        self.metric = metric
        self.index_ = None
        self.scale_ = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "HNSWFamiliarity":
        """Build HNSW index."""
        X = np.asarray(X).astype(np.float32)

        try:
            import hnswlib
        except ImportError:
            raise ImportError("hnswlib required. Install: pip install hnswlib")

        dim = X.shape[1]
        self.index_ = hnswlib.Index(space=self.metric, dim=dim)
        self.index_.init_index(max_elements=len(X), ef_construction=self.ef_construction, M=self.M)
        self.index_.add_items(X)
        self.index_.set_ef(self.ef_search)

        # Compute scale from k-th neighbor distances
        labels, distances = self.index_.knn_query(X, k=self.k + 1)
        self.scale_ = np.median(distances[:, -1]) + 1e-8

        self.is_fitted_ = True
        return self

    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Compute familiarity scores.

        Returns
        -------
        scores : ndarray
            Familiarity in [0, 1].
        """
        self._check_is_fitted()
        X = np.asarray(X).astype(np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        labels, distances = self.index_.knn_query(X, k=1)
        scores = np.exp(-distances.squeeze() / self.scale_)
        return scores

    def batch_query(self, X: npt.ArrayLike, batch_size: int = 1000) -> npt.NDArray:
        """Process large batches efficiently."""
        X = np.asarray(X)
        results = []
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            scores = self.score(batch)
            results.append(scores)
        return np.concatenate(results)
