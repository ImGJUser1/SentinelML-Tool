import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/deep_learning/uncertainty/deep_ensembles.py
"""
Deep Ensemble uncertainty estimation.
"""

from typing import Callable, List, Optional

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseTrustModel


class DeepEnsembleUncertainty(BaseTrustModel):
    """
    Uncertainty quantification using deep ensembles.

    Multiple models trained with different initializations
    provide predictive mean and epistemic uncertainty.

    Parameters
    ----------
    models : list of callables
        List of trained models.
    method : str, default='variance'
        Uncertainty aggregation ('variance', 'entropy', 'mutual_info').

    Examples
    --------
    >>> models = [create_model() for _ in range(5)]
    >>> for m in models:
    ...     m.fit(X_train, y_train)
    >>> ensemble = DeepEnsembleUncertainty(models)
    >>> trust_scores = ensemble.score(X_test)
    """

    def __init__(
        self,
        models: List[Callable],
        name: str = "DeepEnsembleUncertainty",
        calibration_method: str = "isotonic",
        method: str = "variance",
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.models = models
        self.method = method
        self._is_torch = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "DeepEnsembleUncertainty":
        """No fitting needed (models already trained)."""
        self.is_fitted_ = True
        return self

    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Compute trust from ensemble disagreement.

        Returns
        -------
        scores : ndarray
            Trust scores in [0, 1].
        """
        self._check_is_fitted()
        X = np.asarray(X)

        # Collect predictions from all models
        predictions = []
        for model in self.models:
            pred = self._predict(model, X)
            predictions.append(pred)

        predictions = np.array(predictions)  # (n_models, n_samples, n_classes)

        if self.method == "variance":
            # Variance of predictions
            mean_pred = predictions.mean(axis=0)
            variance = predictions.var(axis=0).mean(axis=-1)
            trust = 1 - np.clip(variance * 4, 0, 1)  # Scale to [0, 1]

        elif self.method == "entropy":
            # Predictive entropy
            mean_pred = predictions.mean(axis=0)
            entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=-1)
            max_entropy = np.log(predictions.shape[-1])
            trust = 1 - entropy / max_entropy

        elif self.method == "mutual_info":
            # Mutual information (epistemic uncertainty)
            mean_pred = predictions.mean(axis=0)
            predictive_entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=-1)

            # Expected entropy
            expected_entropy = -np.mean(
                np.sum(predictions * np.log(predictions + 1e-10), axis=-1), axis=0
            )

            mutual_info = predictive_entropy - expected_entropy
            trust = 1 - np.clip(mutual_info / np.log(predictions.shape[-1]), 0, 1)

        return np.clip(trust, 0, 1)

    def _predict(self, model: Callable, X: npt.NDArray) -> npt.NDArray:
        """Get predictions from model."""
        if self._is_torch is None:
            self._is_torch = self._check_is_torch(model)

        if self._is_torch:
            import torch

            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                out = model(X_tensor)
                return torch.softmax(out, dim=-1).cpu().numpy()
        else:
            # Assume TensorFlow or sklearn-like
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            else:
                return model.predict(X)

    def _check_is_torch(self, model) -> bool:
        try:
            import torch.nn as nn

            return isinstance(model, nn.Module)
        except ImportError:
            return False

    def get_predictions(self, X: npt.ArrayLike) -> npt.NDArray:
        """Get all ensemble predictions."""
        X = np.asarray(X)
        predictions = []
        for model in self.models:
            pred = self._predict(model, X)
            predictions.append(pred)
        return np.array(predictions)
