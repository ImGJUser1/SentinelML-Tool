import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/deep_learning/feature_drift/activation_monitor.py
"""
Monitor layer-wise activations for feature drift.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseDetector


class ActivationMonitor(BaseDetector):
    """
    Monitor intermediate layer activations for drift.

    Detects when internal representations shift, indicating
    covariate shift at the feature level.

    Parameters
    ----------
    model : callable
        Neural network with accessible layers.
    layer_names : list of str
        Layers to monitor.
    detector_factory : callable
        Function creating drift detectors for each layer.

    Examples
    --------
    >>> monitor = ActivationMonitor(
    ...     model=resnet,
    ...     layer_names=['layer1', 'layer2', 'layer3'],
    ...     detector_factory=lambda: MMDDriftDetector()
    ... )
    >>> monitor.fit(X_reference)
    >>> is_drift, scores = monitor.detect(X_new)
    """

    def __init__(
        self,
        model: Callable,
        layer_names: List[str],
        detector_factory: Callable[[], BaseDetector],
        name: str = "ActivationMonitor",
        threshold: float = 0.05,
        aggregation: str = "any",
        verbose: bool = False,
    ):
        super().__init__(name=name, threshold=threshold, verbose=verbose)
        self.model = model
        self.layer_names = layer_names
        self.detector_factory = detector_factory
        self.aggregation = aggregation
        self.activation_detectors_: Dict[str, BaseDetector] = {}
        self._activation_hooks: Dict[str, Any] = {}
        self._is_torch = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "ActivationMonitor":
        """Extract reference activations and fit detectors."""
        X = np.asarray(X)

        # Determine framework
        self._is_torch = self._check_is_torch()

        # Extract activations
        activations = self._extract_activations(X)

        # Fit detector for each layer
        for layer_name in self.layer_names:
            if layer_name not in activations:
                raise ValueError(f"Layer {layer_name} not found in model")

            detector = self.detector_factory()
            # Flatten activations for drift detection
            act_data = self._flatten_activations(activations[layer_name])
            detector.fit(act_data, y)
            self.activation_detectors_[layer_name] = detector

        self.is_fitted_ = True
        return self

    def detect(self, X: npt.ArrayLike) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """
        Detect drift in layer activations.

        Returns
        -------
        is_drift : ndarray
            Boolean drift flags.
        scores : ndarray
            Aggregated drift scores.
        """
        self._check_is_fitted()
        X = np.asarray(X)

        # Extract new activations
        activations = self._extract_activations(X)

        # Check each layer
        layer_drifts = {}
        layer_scores = {}

        for layer_name, detector in self.activation_detectors_.items():
            act_data = self._flatten_activations(activations[layer_name])
            is_drift, scores = detector.detect(act_data)
            layer_drifts[layer_name] = is_drift
            layer_scores[layer_name] = scores

        # Aggregate across layers
        if self.aggregation == "any":
            # Drift if any layer shows drift
            combined_drift = np.any([d for d in layer_drifts.values()], axis=0)
        elif self.aggregation == "all":
            # Drift only if all layers show drift
            combined_drift = np.all([d for d in layer_drifts.values()], axis=0)
        elif self.aggregation == "mean":
            # Average scores
            combined_drift = np.mean([s for s in layer_scores.values()], axis=0) < self.threshold
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Average p-values as score
        combined_scores = np.mean([s for s in layer_scores.values()], axis=0)

        n_samples = len(X)
        return (
            np.array([combined_drift] * n_samples)
            if np.isscalar(combined_drift)
            else combined_drift,
            combined_scores
            if not np.isscalar(combined_scores)
            else np.array([combined_scores] * n_samples),
        )

    def _extract_activations(self, X: npt.NDArray) -> Dict[str, npt.NDArray]:
        """Extract activations from all monitored layers."""
        if self._is_torch:
            return self._extract_torch_activations(X)
        else:
            return self._extract_tf_activations(X)

    def _extract_torch_activations(self, X: npt.NDArray) -> Dict[str, npt.NDArray]:
        """PyTorch activation extraction."""
        import torch

        activations = {}
        handles = []

        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach().cpu().numpy()

            return hook

        # Register hooks
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                handle = module.register_forward_hook(hook_fn(name))
                handles.append(handle)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            self.model(X)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return activations

    def _extract_tf_activations(self, X: npt.NDArray) -> Dict[str, npt.NDArray]:
        """TensorFlow activation extraction."""
        import tensorflow as tf

        # Create intermediate model
        layer_outputs = []
        layer_names_found = []

        for layer in self.model.layers:
            if layer.name in self.layer_names:
                layer_outputs.append(layer.output)
                layer_names_found.append(layer.name)

        if not layer_outputs:
            raise ValueError(f"Layers {self.layer_names} not found in model")

        activation_model = tf.keras.Model(inputs=self.model.input, outputs=layer_outputs)

        outputs = activation_model.predict(X, verbose=0)
        if not isinstance(outputs, list):
            outputs = [outputs]

        return {name: out for name, out in zip(layer_names_found, outputs)}

    def _flatten_activations(self, activation: npt.NDArray) -> npt.NDArray:
        """Flatten activation tensors to 2D."""
        if activation.ndim > 2:
            # For conv layers: flatten spatial dimensions
            return activation.reshape(activation.shape[0], -1)
        return activation

    def _check_is_torch(self) -> bool:
        try:
            import torch.nn as nn

            return isinstance(self.model, nn.Module)
        except ImportError:
            return False

    def get_layer_importance(self) -> Dict[str, float]:
        """Return drift sensitivity per layer."""
        return {
            name: detector.threshold if hasattr(detector, "threshold") else 0.05
            for name, detector in self.activation_detectors_.items()
        }
