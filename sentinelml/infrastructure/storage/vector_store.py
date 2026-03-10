import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/infrastructure/storage/vector_store.py
"""
Vector store integration for reference data.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseSentinelComponent


class VectorStore(BaseSentinelComponent):
    """
    Interface to vector databases for reference storage.

    Supports multiple backends: FAISS, Chroma, Pinecone, Weaviate.

    Parameters
    ----------
    backend : str, default='faiss'
        Vector store backend.
    embedding_function : callable
        Function to generate embeddings.
    dimension : int, optional
        Embedding dimension.

    Examples
    --------
    >>> store = VectorStore(
    ...     backend='faiss',
    ...     embedding_function=sentence_transformer.encode,
    ...     dimension=384
    ... )
    >>> store.add(documents, ids)
    >>> results = store.search(query, k=5)
    """

    def __init__(
        self,
        backend: str = "faiss",
        embedding_function: Optional[Callable] = None,
        name: str = "VectorStore",
        dimension: Optional[int] = None,
        index_name: str = "sentinelml_index",
        verbose: bool = False,
        **backend_kwargs,
    ):
        super().__init__(name=name, verbose=verbose)
        self.backend = backend
        self.embedding_function = embedding_function
        self.dimension = dimension
        self.index_name = index_name
        self.backend_kwargs = backend_kwargs

        self._store = None
        self._ids: List[str] = []

    def fit(self, X=None, y=None):
        """Initialize vector store."""
        if self.backend == "faiss":
            self._init_faiss()
        elif self.backend == "chroma":
            self._init_chroma()
        elif self.backend == "pinecone":
            self._init_pinecone()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        self.is_fitted_ = True
        return self

    def _init_faiss(self):
        """Initialize FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu or faiss-gpu required")

        if self.dimension is None:
            raise ValueError("dimension required for FAISS")

        # Create index
        self._store = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine if normalized)

        if self.verbose:
            print(f"Initialized FAISS index (dim={self.dimension})")

    def _init_chroma(self):
        """Initialize Chroma client."""
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb required")

        client = chromadb.Client()

        self._store = client.get_or_create_collection(
            name=self.index_name, embedding_function=self.embedding_function
        )

        if self.verbose:
            print(f"Initialized Chroma collection: {self.index_name}")

    def _init_pinecone(self):
        """Initialize Pinecone index."""
        try:
            import pinecone
        except ImportError:
            raise ImportError("pinecone-client required")

        pinecone.init(**self.backend_kwargs)
        self._store = pinecone.Index(self.index_name)

        if self.verbose:
            print(f"Initialized Pinecone index: {self.index_name}")

    def add(
        self,
        documents: List[str],
        ids: Optional[List[str]] = None,
        embeddings: Optional[npt.NDArray] = None,
        metadatas: Optional[List[Dict]] = None,
    ):
        """
        Add documents to store.

        Parameters
        ----------
        documents : list of str
            Documents to add.
        ids : list of str, optional
            Document IDs (auto-generated if not provided).
        embeddings : ndarray, optional
            Pre-computed embeddings.
        metadatas : list of dict, optional
            Metadata for each document.
        """
        if ids is None:
            import uuid

            ids = [str(uuid.uuid4()) for _ in documents]

        self._ids.extend(ids)

        # Generate embeddings if not provided
        if embeddings is None and self.embedding_function is not None:
            embeddings = np.array([self.embedding_function(d) for d in documents])

        if self.backend == "faiss":
            if embeddings is not None:
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings)
                self._store.add(embeddings.astype("float32"))
        elif self.backend == "chroma":
            self._store.add(
                documents=documents,
                ids=ids,
                embeddings=embeddings.tolist() if embeddings is not None else None,
                metadatas=metadatas,
            )
        elif self.backend == "pinecone":
            vectors = [
                (id, emb.tolist(), meta or {})
                for id, emb, meta in zip(ids, embeddings, metadatas or [{}] * len(ids))
            ]
            self._store.upsert(vectors=vectors)

        if self.verbose:
            print(f"Added {len(documents)} documents")

    def search(
        self, query: str, k: int = 5, filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Parameters
        ----------
        query : str
            Query string.
        k : int
            Number of results.
        filter_dict : dict, optional
            Metadata filter.

        Returns
        -------
        list of results with 'id', 'document', 'score'.
        """
        # Generate query embedding
        if self.embedding_function is None:
            raise ValueError("embedding_function required for search")

        query_emb = self.embedding_function(query).reshape(1, -1)

        if self.backend == "faiss":
            faiss.normalize_L2(query_emb.astype("float32"))
            scores, indices = self._store.search(query_emb.astype("float32"), k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self._ids):
                    results.append({"id": self._ids[idx], "score": float(score), "index": int(idx)})
            return results

        elif self.backend == "chroma":
            results = self._store.query(
                query_embeddings=query_emb.tolist(), n_results=k, where=filter_dict
            )

            return [
                {"id": id, "document": doc, "score": dist, "metadata": meta}
                for id, doc, dist, meta in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["distances"][0],
                    results["metadatas"][0] if results["metadatas"] else [{}] * k,
                )
            ]

        elif self.backend == "pinecone":
            results = self._store.query(vector=query_emb[0].tolist(), top_k=k, filter=filter_dict)

            return [
                {"id": match["id"], "score": match["score"], "metadata": match.get("metadata", {})}
                for match in results["matches"]
            ]

        return []

    def delete(self, ids: List[str]):
        """Delete documents by ID."""
        if self.backend == "faiss":
            # FAISS doesn't support deletion, mark as deleted
            for id in ids:
                if id in self._ids:
                    self._ids[self._ids.index(id)] = None
        elif self.backend == "chroma":
            self._store.delete(ids=ids)
        elif self.backend == "pinecone":
            self._store.delete(ids=ids)

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "backend": self.backend,
            "index_name": self.index_name,
            "n_documents": len([id for id in self._ids if id is not None]),
            "dimension": self.dimension,
        }
