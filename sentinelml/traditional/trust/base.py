import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/traditional/trust/base.py
"""
Base class for trust models.
"""

from abc import abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseTrustModel


class BaseTraditionalTrust(BaseTrustModel):
    """
    Base class for traditional ML trust models.

    Provides common functionality for statistical
    and distance-based trust scoring.

    Parameters
    ----------
    calibration_method : str, default='isotonic'
        Method for calibrating trust scores.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        calibration_method: str = "isotonic",
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)

    @abstractmethod
    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Compute trust scores for input data.

        Returns
        -------
        scores : ndarray
            Trust scores in [0, 1].
        """
        pass
