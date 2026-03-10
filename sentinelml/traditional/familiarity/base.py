import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/traditional/familiarity/base.py
"""
Base class for familiarity models.
"""

from abc import abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseTrustModel


class BaseFamiliarity(BaseTrustModel):
    """
    Base class for familiarity/density estimation models.

    Familiarity measures how similar samples are to
    the reference distribution.

    Parameters
    ----------
    metric : str, default='euclidean'
        Distance metric for similarity.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        metric: str = "euclidean",
        calibration_method: str = "isotonic",
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.metric = metric
        self.reference_data_: Optional[npt.NDArray] = None

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "BaseFamiliarity":
        """Fit familiarity model to reference data."""
        pass

    @abstractmethod
    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Compute familiarity scores.

        Returns
        -------
        scores : ndarray
            Familiarity in [0, 1], higher = more familiar.
        """
        pass
