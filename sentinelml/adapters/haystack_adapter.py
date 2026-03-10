import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/adapters/haystack_adapter.py
"""
Haystack integration adapter.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseSentinelComponent


class HaystackAdapter(BaseSentinelComponent):
    """
    Adapter for Haystack pipelines.

    Monitors Haystack pipelines with trust scoring
    at retrieval and generation stages.

    Parameters
    ----------
    pipeline : object
        Haystack Pipeline object.
    monitoring_nodes : list
        Node names to monitor.

    Examples
    --------
    >>> from haystack import Pipeline
    >>> pipeline = Pipeline()
    >>> # ... add nodes ...
    >>> adapter = HaystackAdapter(pipeline, monitoring_nodes=['Retriever', 'Generator'])
    >>> result = adapter.run(query="Question?", trust_check=True)
    """

    def __init__(
        self,
        pipeline: Any,
        name: str = "HaystackAdapter",
        monitoring_nodes: Optional[List[str]] = None,
        guardrails: Optional[Dict[str, List[Any]]] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.pipeline = pipeline
        self.monitoring_nodes = monitoring_nodes or ["Retriever", "Generator"]
        self.guardrails = guardrails or {}

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def run(
        self, query: str, params: Optional[Dict] = None, trust_check: bool = True
    ) -> Dict[str, Any]:
        """
        Run pipeline with monitoring.

        Parameters
        ----------
        query : str
            Query string.
        params : dict
            Pipeline parameters.
        trust_check : bool
            Whether to run trust checks.

        Returns
        -------
        dict with pipeline output and trust information.
        """
        # Check input
        if trust_check and "input" in self.guardrails:
            for guardrail in self.guardrails["input"]:
                result = guardrail.validate(query)
                if not result["is_valid"] and result["action"] == "block":
                    return {"answers": [], "error": "Query blocked", "guardrail_result": result}

        # Run pipeline
        try:
            result = self.pipeline.run(query=query, params=params or {})
        except Exception as e:
            return {"answers": [], "error": str(e)}

        # Extract answers
        answers = result.get("answers", [])
        answer_texts = [str(a.answer) for a in answers]

        # Check outputs
        guardrail_results = []
        if trust_check and "output" in self.guardrails:
            for answer_text in answer_texts:
                for guardrail in self.guardrails["output"]:
                    gr_result = guardrail.validate(
                        answer_text,
                        context={"query": query, "documents": result.get("documents", [])},
                    )
                    guardrail_results.append(gr_result)

        return {
            "answers": answer_texts,
            "documents": [
                {"content": d.content[:500], "score": d.score} for d in result.get("documents", [])
            ],
            "guardrail_results": guardrail_results if trust_check else None,
        }

    def add_guardrail(self, node: str, guardrail: Any):
        """Add guardrail for specific node."""
        if node not in self.guardrails:
            self.guardrails[node] = []
        self.guardrails[node].append(guardrail)

    def get_pipeline(self) -> Any:
        """Return underlying Haystack pipeline."""
        return self.pipeline
