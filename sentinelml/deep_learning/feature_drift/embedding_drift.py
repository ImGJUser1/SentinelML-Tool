from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/deep_learning/feature_drift/embedding_drift.py
"""
Drift detection in embedding/semantic space.
"""

from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

from sentinelml.traditional.drift.mmd_detector import MMDDriftDetector


class EmbeddingDriftDetector(MMDDriftDetector):
    """
    Specialized drift detector for high-dimensional embeddings.

    Uses dimensionality reduction before MMD for efficiency,
    and handles cosine similarity for semantic embeddings.

    Parameters
    ----------
    embedding_model : callable, optional
        Model to generate embeddings. If None, assumes input is embeddings.
    reduction_method : str, default='pca'
        Dimensionality reduction ('pca', 'umap', 'none').
    n_components : int, default=50
        Target dimensions after reduction.
    use_cosine : bool, default=True
        Use cosine similarity for embeddings.

    Examples
    --------
    >>> detector = EmbeddingDriftDetector(
    ...     embedding_model=sentence_transformer,
    ...     reduction_method='pca',
    ...     n_components=100
    ... )
    >>> detector.fit(text_reference)
    >>> is_drift, scores = detector.detect(text_new)
    """

    def __init__(
        self,
        name: str = "EmbeddingDriftDetector",
        embedding_model: Optional[Callable] = None,
        reduction_method: str = "pca",
        n_components: int = 50,
        use_cosine: bool = True,
        window_size: int = 1000,
        threshold: float = 0.05,
        verbose: bool = False,
    ):
        # Initialize with RBF kernel (will be overridden if cosine)
        kernel = "linear" if use_cosine else "rbf"
        super().__init__(
            name=name, window_size=window_size, threshold=threshold, kernel=kernel, verbose=verbose
        )
        self.embedding_model = embedding_model
        self.reduction_method = reduction_method
        self.n_components = n_components
        self.use_cosine = use_cosine
        self.reducer_ = None
        self.reference_embeddings_ = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "EmbeddingDriftDetector":
        """Generate embeddings and fit drift detector."""
        # Generate embeddings if model provided
        if self.embedding_model is not None:
            X = self._get_embeddings(X)
        else:
            X = np.asarray(X)

        self.reference_embeddings_ = X.copy()

        # Dimensionality reduction
        if self.reduction_method == "pca":
            from sklearn.decomposition import PCA

            n_comp = min(self.n_components, X.shape[0], X.shape[1])
            self.reducer_ = PCA(n_components=n_comp)
            X_reduced = self.reducer_.fit_transform(X)
        elif self.reduction_method == "umap":
            try:
                import umap

                self.reducer_ = umap.UMAP(n_components=self.n_components)
                X_reduced = self.reducer_.fit_transform(X)
            except ImportError:
                raise ImportError("umap-learn required for UMAP reduction")
        else:
            X_reduced = X

        # Normalize for cosine similarity
        if self.use_cosine:
            X_reduced = X_reduced / (np.linalg.norm(X_reduced, axis=1, keepdims=True) + 1e-8)

        # Fit MMD on reduced space
        super().fit(X_reduced, y)
        return self

    def detect(self, X: npt.ArrayLike) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """Detect drift in embedding space."""
        # Generate embeddings if needed
        if self.embedding_model is not None:
            X = self._get_embeddings(X)
        else:
            X = np.asarray(X)

        # Apply reduction
        if self.reducer_ is not None:
            X_reduced = self.reducer_.transform(X)
        else:
            X_reduced = X

        # Normalize
        if self.use_cosine:
            X_reduced = X_reduced / (np.linalg.norm(X_reduced, axis=1, keepdims=True) + 1e-8)

        return super().detect(X_reduced)

    def _get_embeddings(self, texts: npt.ArrayLike) -> npt.NDArray:
        """Generate embeddings from texts."""
        if isinstance(texts, np.ndarray) and texts.ndim > 1:
            # Already embeddings
            return texts

        # Assume list of strings or similar
        if hasattr(self.embedding_model, "encode"):
            # Sentence-transformers style
            return self.embedding_model.encode(texts)
        else:
            # Generic callable
            return self.embedding_model(texts)

    def fit_transform(self, X: npt.ArrayLike) -> npt.NDArray:
        """Fit and return embeddings."""
        self.fit(X)
        return self.reference_embeddings_
