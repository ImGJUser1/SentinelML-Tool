import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/deep_learning/uncertainty/base.py
"""
Base class for deep learning uncertainty methods.
"""

from abc import abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseTrustModel


class BaseUncertainty(BaseTrustModel):
    """
    Base class for deep learning uncertainty quantification.

    Provides interface for model-aware uncertainty methods
    like MC Dropout, Ensembles, and Evidential learning.

    Parameters
    ----------
    model : callable
        Neural network model.
    """

    def __init__(
        self,
        model: Callable,
        name: Optional[str] = None,
        calibration_method: str = "isotonic",
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.model = model
        self._is_torch: Optional[bool] = None

    def _check_is_torch(self, model: Optional[Any] = None) -> bool:
        """Check if model is PyTorch."""
        if model is None:
            model = self.model

        try:
            import torch.nn as nn

            return isinstance(model, nn.Module)
        except ImportError:
            return False

    @abstractmethod
    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Compute uncertainty-based trust scores.

        Returns
        -------
        scores : ndarray
            Trust scores (higher = more certain).
        """
        pass

    def predict_with_uncertainty(self, X: npt.ArrayLike) -> tuple:
        """
        Predict with uncertainty quantification.

        Returns
        -------
        predictions : ndarray
            Model predictions.
        uncertainties : ndarray
            Uncertainty estimates.
        """
        raise NotImplementedError
