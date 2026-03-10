from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/core/base.py
"""
Abstract base classes for all SentinelML components.

Provides the foundational interfaces that ensure consistency across
traditional ML, deep learning, GenAI, RAG, and agentic monitoring.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class BaseSentinelComponent(ABC, BaseEstimator):
    """
    Base class for all SentinelML components.

    Inherits from sklearn.base.BaseEstimator for compatibility with
    scikit-learn pipelines and model selection tools.

    Parameters
    ----------
    name : str, optional
        Component identifier for logging and reporting.
    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(self, name: Optional[str] = None, verbose: bool = False):
        self.name = name or self.__class__.__name__
        self.verbose = verbose
        self.is_fitted_ = False
        self.metadata_: Dict[str, Any] = {}

    def _log(self, message: str, level: int = logging.INFO):
        """Log message if verbose is enabled."""
        if self.verbose:
            logger.log(level, f"[{self.name}] {message}")

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None) -> "BaseSentinelComponent":
        """
        Fit the component to reference data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Reference/training data.
        y : array-like of shape (n_samples,), optional
            Target values (if supervised).

        Returns
        -------
        self : BaseSentinelComponent
            Fitted component.
        """
        pass

    def _check_is_fitted(self):
        """Validate that the component has been fitted."""
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.name} is not fitted. Call fit() before using this component."
            )

    def get_metadata(self) -> Dict[str, Any]:
        """Return component metadata for reporting."""
        return {"name": self.name, "is_fitted": self.is_fitted_, **self.metadata_}


class BaseDetector(BaseSentinelComponent, TransformerMixin):
    """
    Base class for drift and anomaly detectors.

    Detectors identify when data or model behavior deviates from
    the reference distribution.

    Parameters
    ----------
    threshold : float, optional
        Detection threshold. If None, determined during fit.
    """

    def __init__(
        self, name: Optional[str] = None, threshold: Optional[float] = None, verbose: bool = False
    ):
        super().__init__(name=name, verbose=verbose)
        self.threshold = threshold

    @abstractmethod
    def detect(self, X: npt.ArrayLike) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """
        Detect anomalies or drift in input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to assess.

        Returns
        -------
        is_anomalous : ndarray of shape (n_samples,)
            Boolean array indicating detected anomalies.
        scores : ndarray of shape (n_samples,)
            Continuous anomaly/drift scores (higher = more anomalous).
        """
        pass

    def fit_detect(self, X: npt.ArrayLike) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """Fit and then detect on same data."""
        return self.fit(X).detect(X)

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Transform data to anomaly scores (sklearn compatibility).

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores.
        """
        _, scores = self.detect(X)
        return scores.reshape(-1, 1)


class BaseTrustModel(BaseSentinelComponent):
    """
    Base class for trust/reliability scoring models.

    Trust models quantify prediction reliability on a scale [0, 1].

    Parameters
    ----------
    calibration_method : str, default='isotonic'
        Method for calibrating trust scores ('isotonic', 'platt', 'beta').
    """

    def __init__(
        self,
        name: Optional[str] = None,
        calibration_method: str = "isotonic",
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.calibration_method = calibration_method
        self.calibrator_ = None

    @abstractmethod
    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Compute trust scores for input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to assess.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Trust scores in [0, 1] range.
        """
        pass

    def calibrate(self, X: npt.ArrayLike, y_true: npt.ArrayLike, y_pred: npt.ArrayLike):
        """
        Calibrate trust scores using ground truth.

        Parameters
        ----------
        X : array-like
            Input features.
        y_true : array-like
            Ground truth labels.
        y_pred : array-like
            Model predictions.
        """
        from sklearn.calibration import CalibratedClassifierCV

        scores = self.score(X)
        # Fit calibration model
        self._fit_calibration(scores, y_true == y_pred)

    def _fit_calibration(self, scores: npt.NDArray, correct: npt.NDArray[np.bool_]):
        """Fit calibration model mapping scores to accuracy."""
        from sklearn.isotonic import IsotonicRegression

        if self.calibration_method == "isotonic":
            self.calibrator_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.calibrator_.fit(scores, correct)

    def score_calibrated(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return calibrated trust scores."""
        scores = self.score(X)
        if self.calibrator_ is not None:
            return self.calibrator_.predict(scores)
        return scores


class BaseGuardrail(BaseSentinelComponent):
    """
    Base class for input/output guardrails (GenAI focus).

    Guardrails validate and filter content in LLM pipelines.

    Parameters
    ----------
    fail_mode : str, default='filter'
        Behavior on validation failure ('filter', 'flag', 'block').
    """

    VALID_FAIL_MODES = ["filter", "flag", "block", "sanitize"]

    def __init__(
        self, name: Optional[str] = None, fail_mode: str = "filter", verbose: bool = False
    ):
        super().__init__(name=name, verbose=verbose)
        if fail_mode not in self.VALID_FAIL_MODES:
            raise ValueError(f"fail_mode must be one of {self.VALID_FAIL_MODES}")
        self.fail_mode = fail_mode

    @abstractmethod
    def validate(self, content: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate content against guardrail rules.

        Parameters
        ----------
        content : Any
            Content to validate (text, structured data, etc.).
        context : dict, optional
            Additional context for validation.

        Returns
        -------
        result : dict
            Validation result with keys:
            - 'is_valid': bool
            - 'score': float (confidence)
            - 'metadata': dict (detailed info)
            - 'action': str ('pass', 'filter', 'block')
        """
        pass

    def __call__(self, content: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Make guardrail callable."""
        return self.validate(content, context)


class BaseValidator(ABC):
    """
    Lightweight validation utility for data preprocessing.

    Not a full component—used for input sanitization.
    """

    @staticmethod
    def check_array(
        X: Any, accept_sparse: bool = False, dtype: Optional[str] = None
    ) -> npt.NDArray:
        """Validate and convert to numpy array."""
        from sklearn.utils.validation import check_array as sklearn_check_array

        return sklearn_check_array(X, accept_sparse=accept_sparse, dtype=dtype)

    @staticmethod
    def check_non_negative(X: npt.ArrayLike, whom: str = "Input") -> npt.NDArray:
        """Ensure array contains no negative values."""
        X = np.asarray(X)
        if np.any(X < 0):
            raise ValueError(f"{whom} must contain only non-negative values")
        return X
