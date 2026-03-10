import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/traditional/drift/base.py
"""
Base class for drift detectors.
"""

from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseDetector


class BaseDriftDetector(BaseDetector):
    """
    Base class for all drift detectors.

    Parameters
    ----------
    reference_data : array-like, optional
        Initial reference distribution.
    window_size : int, default=1000
        Size of sliding window for online detection.
    threshold : float, default=0.05
        Significance level for drift detection.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        window_size: int = 1000,
        threshold: float = 0.05,
        verbose: bool = False,
    ):
        super().__init__(name=name, threshold=threshold, verbose=verbose)
        self.window_size = window_size
        self.reference_data_: Optional[npt.NDArray] = None
        self.current_window_: list = []

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "BaseDriftDetector":
        """Store reference distribution."""
        X = np.asarray(X)
        self.reference_data_ = X.copy()
        self.is_fitted_ = True
        return self

    def update(self, X: npt.ArrayLike):
        """Update sliding window with new data."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.current_window_.extend(X)
        if len(self.current_window_) > self.window_size:
            self.current_window_ = self.current_window_[-self.window_size :]

    @abstractmethod
    def detect(self, X: npt.ArrayLike) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """
        Detect drift in input data.

        Returns
        -------
        is_drift : ndarray
            Boolean flags per sample.
        p_values : ndarray
            Statistical significance scores.
        """
        pass

    def detect_window(self) -> Tuple[bool, float]:
        """Detect drift in current window vs reference."""
        if len(self.current_window_) < 30:
            return False, 1.0

        window_array = np.array(self.current_window_)
        is_drift, p_values = self.detect(window_array)
        return bool(is_drift.any()), float(np.mean(p_values))
