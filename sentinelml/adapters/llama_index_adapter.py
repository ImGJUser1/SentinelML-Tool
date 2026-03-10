import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/adapters/llama_index_adapter.py
"""
LlamaIndex integration adapter.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseSentinelComponent


class LlamaIndexAdapter(BaseSentinelComponent):
    """
    Adapter for LlamaIndex RAG pipelines.

    Monitors retrieval and generation in LlamaIndex
    with trust scoring and guardrails.

    Parameters
    ----------
    index : object
        LlamaIndex index.
    query_engine : object, optional
        Query engine (created from index if not provided).

    Examples
    --------
    >>> from llama_index import VectorStoreIndex
    >>> index = VectorStoreIndex.from_documents(docs)
    >>> adapter = LlamaIndexAdapter(index)
    >>> result = adapter.query("Question?", trust_check=True)
    """

    def __init__(
        self,
        index: Any,
        query_engine: Optional[Any] = None,
        name: str = "LlamaIndexAdapter",
        retrieval_guardrails: Optional[List[Any]] = None,
        generation_guardrails: Optional[List[Any]] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.index = index
        self.query_engine = query_engine or index.as_query_engine()
        self.retrieval_guardrails = retrieval_guardrails or []
        self.generation_guardrails = generation_guardrails or []

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def query(
        self, query_str: str, trust_check: bool = True, return_source_nodes: bool = False
    ) -> Dict[str, Any]:
        """
        Query index with monitoring.

        Parameters
        ----------
        query_str : str
            Query string.
        trust_check : bool
            Whether to run trust checks.
        return_source_nodes : bool
            Whether to return retrieved nodes.

        Returns
        -------
        dict with response and trust information.
        """
        # Check query
        if trust_check:
            for guardrail in self.retrieval_guardrails:
                result = guardrail.validate(query_str)
                if not result["is_valid"] and result["action"] == "block":
                    return {
                        "response": None,
                        "error": "Query blocked by guardrail",
                        "guardrail_result": result,
                    }

        # Execute query
        try:
            response = self.query_engine.query(query_str)
        except Exception as e:
            return {"response": None, "error": str(e)}

        # Extract response text
        response_text = str(response)

        # Check response
        guardrail_results = []
        if trust_check:
            for guardrail in self.generation_guardrails:
                result = guardrail.validate(
                    response_text,
                    context={"query": query_str, "source_nodes": response.source_nodes},
                )
                guardrail_results.append(result)

        result = {
            "response": response_text,
            "guardrail_results": guardrail_results if trust_check else None,
        }

        if return_source_nodes:
            result["source_nodes"] = [
                {"text": node.node.text[:500], "score": node.score, "metadata": node.node.metadata}
                for node in response.source_nodes
            ]

        return result

    def retrieve(self, query_str: str) -> List[Dict[str, Any]]:
        """Retrieve nodes without generation."""
        retriever = self.index.as_retriever()
        nodes = retriever.retrieve(query_str)

        return [
            {"text": node.node.text[:500], "score": node.score, "metadata": node.node.metadata}
            for node in nodes
        ]

    def add_retrieval_guardrail(self, guardrail: Any):
        """Add guardrail for retrieval phase."""
        self.retrieval_guardrails.append(guardrail)

    def add_generation_guardrail(self, guardrail: Any):
        """Add guardrail for generation phase."""
        self.generation_guardrails.append(guardrail)

    def get_index(self) -> Any:
        """Return underlying LlamaIndex index."""
        return self.index
