import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/adapters/tensorflow_adapter.py
"""
TensorFlow/Keras model adapter.
"""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseSentinelComponent


class TensorflowAdapter(BaseSentinelComponent):
    """
    Adapter for TensorFlow/Keras models.

    Provides unified interface for TF models
    with automatic batching and output handling.

    Parameters
    ----------
    model : tf.keras.Model
        TensorFlow model.
    batch_size : int, default=32
        Inference batch size.

    Examples
    --------
    >>> model = create_tf_model()
    >>> adapter = TensorflowAdapter(model)
    >>> predictions = adapter.predict(X_test)
    """

    def __init__(
        self,
        model: Any,
        name: str = "TensorflowAdapter",
        batch_size: int = 32,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.model = model
        self.batch_size = batch_size

    def fit(self, X=None, y=None):
        """No fitting (model should be pre-trained)."""
        self.is_fitted_ = True
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray:
        """
        Generate predictions.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        predictions : ndarray
            Model predictions.
        """
        X = np.asarray(X)

        # Use model's predict with batch size
        predictions = self.model.predict(X, batch_size=self.batch_size, verbose=0)

        # Ensure numpy array
        if hasattr(predictions, "numpy"):
            predictions = predictions.numpy()

        return np.asarray(predictions)

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray:
        """Get probability predictions."""
        preds = self.predict(X)

        # If already probabilities (sum to ~1)
        if preds.ndim == 2 and np.allclose(preds.sum(axis=1), 1.0, rtol=0.1):
            return preds

        # Apply softmax
        if preds.ndim == 2:
            exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
            probs = exp_preds / exp_preds.sum(axis=1, keepdims=True)
            return probs

        # Binary case
        sigmoid = 1 / (1 + np.exp(-preds))
        return np.column_stack([1 - sigmoid, sigmoid])

    def get_embeddings(self, X: npt.ArrayLike, layer_name: Optional[str] = None) -> npt.NDArray:
        """
        Extract embeddings from intermediate layer.

        Parameters
        ----------
        X : array-like
            Input data.
        layer_name : str, optional
            Name of layer to extract from. If None, uses penultimate layer.
        """
        import tensorflow as tf

        X = np.asarray(X)

        if layer_name is None:
            # Use penultimate layer
            layer = self.model.layers[-2]
        else:
            layer = self.model.get_layer(layer_name)

        # Create feature extraction model
        feature_model = tf.keras.Model(inputs=self.model.input, outputs=layer.output)

        embeddings = feature_model.predict(X, batch_size=self.batch_size, verbose=0)

        # Flatten if needed
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)

        return embeddings

    def get_model(self) -> Any:
        """Return underlying TensorFlow model."""
        return self.model
