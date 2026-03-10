import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/adapters/torch_adapter.py
"""
PyTorch model adapter.
"""

from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseSentinelComponent


class TorchAdapter(BaseSentinelComponent):
    """
    Adapter for PyTorch models.

    Provides unified interface for PyTorch models
    with automatic device handling and batching.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.
    device : str, default='auto'
        Device to use ('cpu', 'cuda', 'auto').
    batch_size : int, default=32
        Inference batch size.

    Examples
    --------
    >>> model = create_torch_model()
    >>> adapter = TorchAdapter(model, device='cuda')
    >>> predictions = adapter.predict(X_test)
    """

    def __init__(
        self,
        model: Any,
        name: str = "TorchAdapter",
        device: str = "auto",
        batch_size: int = 32,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.model = model
        self.device = self._resolve_device(device)
        self.batch_size = batch_size

        # Move model to device
        if hasattr(model, "to"):
            self.model = model.to(self.device)
            self.model.eval()

    def _resolve_device(self, device: str) -> str:
        """Resolve device string."""
        if device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

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
        import torch

        X = np.asarray(X)
        predictions = []

        # Process in batches
        for i in range(0, len(X), self.batch_size):
            batch = X[i : i + self.batch_size]

            # Convert to tensor
            if not isinstance(batch, torch.Tensor):
                batch_tensor = torch.tensor(batch, dtype=torch.float32, device=self.device)
            else:
                batch_tensor = batch.to(self.device)

            # Forward pass
            with torch.no_grad():
                output = self.model(batch_tensor)

                # Handle different output types
                if isinstance(output, tuple):
                    output = output[0]

                # Move to CPU and convert
                output_np = output.cpu().numpy()
                predictions.append(output_np)

        return np.concatenate(predictions, axis=0)

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray:
        """Get probability predictions."""
        import torch

        logits = self.predict(X)

        # Apply softmax if needed
        if logits.ndim == 2 and logits.shape[1] > 1:
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            return probs

        # Binary case
        sigmoid = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - sigmoid, sigmoid])

    def get_embeddings(self, X: npt.ArrayLike, layer: Optional[str] = None) -> npt.NDArray:
        """
        Extract embeddings from intermediate layer.

        Parameters
        ----------
        X : array-like
            Input data.
        layer : str, optional
            Layer name to extract from. If None, uses penultimate layer.
        """
        import torch

        X = np.asarray(X)
        embeddings = []

        # Hook to capture intermediate output
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        # Register hook if layer specified
        handle = None
        if layer and hasattr(self.model, layer):
            target_layer = getattr(self.model, layer)
            handle = target_layer.register_forward_hook(get_activation("capture"))

        # Forward pass
        for i in range(0, len(X), self.batch_size):
            batch = X[i : i + self.batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                _ = self.model(batch_tensor)

                if "capture" in activation:
                    emb = activation["capture"].cpu().numpy()
                    # Flatten if needed
                    if emb.ndim > 2:
                        emb = emb.reshape(emb.shape[0], -1)
                    embeddings.append(emb)

        if handle:
            handle.remove()

        return np.concatenate(embeddings, axis=0) if embeddings else np.array([])

    def get_model(self) -> Any:
        """Return underlying PyTorch model."""
        return self.model
