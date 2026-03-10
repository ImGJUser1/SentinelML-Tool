import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/deep_learning/adversarial/fgsm_detector.py
"""
Fast Gradient Method (FGM) adversarial detection.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseTrustModel


class FGMDetector(BaseTrustModel):
    """
    Detect adversarial examples using gradient-based perturbations.

    Computes how easily input can be perturbed to change prediction.
    High sensitivity indicates potential adversarial example.

    Parameters
    ----------
    model : callable
        Differentiable model.
    epsilon : float, default=0.01
        Perturbation magnitude.
    norm : str, default='inf'
        Norm constraint ('inf', '2').
    n_iter : int, default=1
        Iterations (1 = FGSM, >1 = iterative).

    Examples
    --------
    >>> detector = FGMDetector(model, epsilon=0.03)
    >>> detector.fit(X_clean)
    >>> robustness_scores = detector.score(X_test)
    """

    def __init__(
        self,
        model: Callable,
        name: str = "FGMDetector",
        calibration_method: str = "isotonic",
        epsilon: float = 0.01,
        norm: str = "inf",
        n_iter: int = 1,
        verbose: bool = False,
    ):
        super().__init__(name=name, calibration_method=calibration_method, verbose=verbose)
        self.model = model
        self.epsilon = epsilon
        self.norm = norm
        self.n_iter = n_iter
        self._is_torch = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "FGMDetector":
        """Store clean data statistics."""
        X = np.asarray(X)
        self._clean_mean = X.mean(axis=0)
        self._clean_std = X.std(axis=0)
        self.is_fitted_ = True
        return self

    def score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Compute robustness-based trust scores.

        Lower score if input is easily perturbed (potentially adversarial).

        Returns
        -------
        scores : ndarray
            Trust scores in [0, 1].
        """
        self._check_is_fitted()
        X = np.asarray(X)

        if self._is_torch is None:
            self._is_torch = self._check_is_torch()

        if self._is_torch:
            return self._score_torch(X)
        else:
            return self._score_tensorflow(X)

    def _score_torch(self, X: npt.NDArray) -> npt.NDArray[np.float64]:
        """PyTorch FGM implementation."""
        import torch

        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)

        # Get original prediction
        with torch.no_grad():
            orig_output = self.model(X_tensor)
            orig_pred = orig_output.argmax(dim=-1)

        # Compute adversarial perturbation
        X_adv = X_tensor.clone().detach().requires_grad_(True)

        for _ in range(self.n_iter):
            if X_adv.grad is not None:
                X_adv.grad.zero_()

            output = self.model(X_adv)
            loss = torch.nn.functional.cross_entropy(output, orig_pred)
            loss.backward()

            # Compute perturbation
            grad = X_adv.grad.data
            if self.norm == "inf":
                perturbation = self.epsilon * grad.sign()
            elif self.norm == "2":
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1, keepdim=True)
                perturbation = self.epsilon * grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
            else:
                raise ValueError(f"Unknown norm: {self.norm}")

            X_adv = X_adv + perturbation
            X_adv = torch.clamp(X_adv, 0, 1).detach().requires_grad_(True)

        # Check if prediction changed
        with torch.no_grad():
            adv_output = self.model(X_adv)
            adv_pred = adv_output.argmax(dim=-1)

        # Robustness = 1 if prediction unchanged, 0 if changed
        robustness = (orig_pred == adv_pred).float().numpy()

        return robustness

    def _score_tensorflow(self, X: npt.NDArray) -> npt.NDArray[np.float64]:
        """TensorFlow FGM implementation."""
        import tensorflow as tf

        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)

        # Get original prediction
        orig_output = self.model(X_tf)
        orig_pred = tf.argmax(orig_output, axis=-1)

        # Create adversarial example
        with tf.GradientTape() as tape:
            tape.watch(X_tf)
            output = self.model(X_tf)
            loss = tf.keras.losses.sparse_categorical_crossentropy(orig_pred, output)

        grad = tape.gradient(loss, X_tf)

        if self.norm == "inf":
            perturbation = self.epsilon * tf.sign(grad)
        elif self.norm == "2":
            grad_norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
            perturbation = self.epsilon * grad / tf.reshape(grad_norm, [-1, 1, 1, 1])

        X_adv = tf.clip_by_value(X_tf + perturbation, 0, 1)

        # Check prediction
        adv_output = self.model(X_adv)
        adv_pred = tf.argmax(adv_output, axis=-1)

        robustness = tf.cast(orig_pred == adv_pred, tf.float32).numpy()
        return robustness

    def _check_is_torch(self) -> bool:
        try:
            import torch.nn as nn

            return isinstance(self.model, nn.Module)
        except ImportError:
            return False

    def generate_adversarial(self, X: npt.ArrayLike) -> npt.NDArray:
        """Generate adversarial examples for testing."""
        self._check_is_fitted()
        X = np.asarray(X)

        if self._is_torch is None:
            self._is_torch = self._check_is_torch()

        if self._is_torch:
            import torch

            self.model.eval()
            X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
            X_adv = X_tensor.clone().detach().requires_grad_(True)

            for _ in range(self.n_iter):
                if X_adv.grad is not None:
                    X_adv.grad.zero_()
                output = self.model(X_adv)
                loss = output[:, output.argmax(dim=-1)].sum()
                loss.backward()
                grad = X_adv.grad.data
                perturbation = self.epsilon * grad.sign()
                X_adv = X_adv + perturbation
                X_adv = torch.clamp(X_adv, 0, 1).detach().requires_grad_(True)

            return X_adv.numpy()
        else:
            import tensorflow as tf

            X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(X_tf)
                output = self.model(X_tf)
                loss = tf.reduce_sum(output[:, tf.argmax(output, axis=-1)])
            grad = tape.gradient(loss, X_tf)
            perturbation = self.epsilon * tf.sign(grad)
            X_adv = tf.clip_by_value(X_tf + perturbation, 0, 1)
            return X_adv.numpy()
