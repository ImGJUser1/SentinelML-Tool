from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/deep_learning/adversarial/boundary_attack.py
"""
Projected Gradient Descent (PGD) adversarial detection.
"""

from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

from sentinelml.deep_learning.adversarial.fgsm_detector import FGMDetector


class PGDDetector(FGMDetector):
    """
    Stronger adversarial detection using PGD attacks.

    PGD is a multi-step adversarial attack that finds
    stronger perturbations than single-step FGM.

    Parameters
    ----------
    epsilon : float, default=0.03
        Maximum perturbation.
    alpha : float, default=0.01
        Step size per iteration.
    n_iter : int, default=40
        Number of attack iterations.
    random_start : bool, default=True
        Use random initialization.

    Examples
    --------
    >>> detector = PGDDetector(model, epsilon=0.03, n_iter=50)
    >>> robustness = detector.score(X_test)
    """

    def __init__(
        self,
        model: Callable,
        name: str = "PGDDetector",
        calibration_method: str = "isotonic",
        epsilon: float = 0.03,
        alpha: float = 0.01,
        n_iter: int = 40,
        norm: str = "inf",
        random_start: bool = True,
        verbose: bool = False,
    ):
        # Initialize parent with n_iter for iterative behavior
        super().__init__(
            model=model,
            name=name,
            calibration_method=calibration_method,
            epsilon=alpha,  # Use alpha as step size
            norm=norm,
            n_iter=n_iter,
            verbose=verbose,
        )
        self.epsilon_total = epsilon
        self.alpha = alpha
        self.random_start = random_start

    def _score_torch(self, X: npt.NDArray) -> npt.NDArray[np.float64]:
        """PyTorch PGD implementation."""
        import torch

        self.model.eval()
        batch_size = X.shape[0]
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Get original predictions
        with torch.no_grad():
            orig_output = self.model(X_tensor)
            orig_pred = orig_output.argmax(dim=-1)

        # Initialize adversarial examples
        if self.random_start:
            delta = torch.empty_like(X_tensor).uniform_(-self.epsilon_total, self.epsilon_total)
            X_adv = torch.clamp(X_tensor + delta, 0, 1).detach()
        else:
            X_adv = X_tensor.clone()

        # PGD iterations
        for _ in range(self.n_iter):
            X_adv.requires_grad_(True)

            output = self.model(X_adv)
            loss = torch.nn.functional.cross_entropy(output, orig_pred)
            loss.backward()

            # Gradient step
            grad = X_adv.grad.data
            if self.norm == "inf":
                X_adv = X_adv + self.alpha * grad.sign()
                # Project to epsilon ball
                delta = torch.clamp(X_adv - X_tensor, -self.epsilon_total, self.epsilon_total)
                X_adv = torch.clamp(X_tensor + delta, 0, 1).detach()
            else:
                # L2 norm
                grad_norm = torch.norm(grad.view(batch_size, -1), dim=1, keepdim=True)
                normalized_grad = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)
                X_adv = X_adv + self.alpha * normalized_grad

                # Project to epsilon ball
                delta = X_adv - X_tensor
                delta_norm = torch.norm(delta.view(batch_size, -1), dim=1, keepdim=True)
                factor = torch.clamp_max(delta_norm, self.epsilon_total) / (delta_norm + 1e-8)
                delta = delta * factor.view(-1, 1, 1, 1)
                X_adv = torch.clamp(X_tensor + delta, 0, 1).detach()

        # Check robustness
        with torch.no_grad():
            adv_output = self.model(X_adv)
            adv_pred = adv_output.argmax(dim=-1)

        robustness = (orig_pred == adv_pred).float().numpy()
        return robustness

    def _score_tensorflow(self, X: npt.NDArray) -> npt.NDArray[np.float64]:
        """TensorFlow PGD implementation."""
        import tensorflow as tf

        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)

        # Original predictions
        orig_output = self.model(X_tf)
        orig_pred = tf.argmax(orig_output, axis=-1)

        # Initialize
        if self.random_start:
            delta = tf.random.uniform(tf.shape(X_tf), -self.epsilon_total, self.epsilon_total)
            X_adv = tf.clip_by_value(X_tf + delta, 0, 1)
        else:
            X_adv = X_tf

        # PGD iterations
        for _ in range(self.n_iter):
            with tf.GradientTape() as tape:
                tape.watch(X_adv)
                output = self.model(X_adv)
                loss = tf.keras.losses.sparse_categorical_crossentropy(orig_pred, output)

            grad = tape.gradient(loss, X_adv)

            if self.norm == "inf":
                X_adv = X_adv + self.alpha * tf.sign(grad)
                delta = tf.clip_by_value(X_adv - X_tf, -self.epsilon_total, self.epsilon_total)
                X_adv = tf.clip_by_value(X_tf + delta, 0, 1)
            else:
                grad_norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
                normalized_grad = grad / tf.reshape(grad_norm, [-1, 1, 1, 1])
                X_adv = X_adv + self.alpha * normalized_grad

                delta = X_adv - X_tf
                delta_norm = tf.norm(tf.reshape(delta, [tf.shape(delta)[0], -1]), axis=1)
                factor = tf.clip_by_value(delta_norm, 0, self.epsilon_total) / (delta_norm + 1e-8)
                delta = delta * tf.reshape(factor, [-1, 1, 1, 1])
                X_adv = tf.clip_by_value(X_tf + delta, 0, 1)

        # Check robustness
        adv_output = self.model(X_adv)
        adv_pred = tf.argmax(adv_output, axis=-1)

        robustness = tf.cast(orig_pred == adv_pred, tf.float32).numpy()
        return robustness
