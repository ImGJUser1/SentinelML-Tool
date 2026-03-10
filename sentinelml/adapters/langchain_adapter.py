import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/adapters/langchain_adapter.py
"""
LangChain integration adapter.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseSentinelComponent


class LangchainAdapter(BaseSentinelComponent):
    """
    Adapter for LangChain components.

    Integrates SentinelML with LangChain chains and agents,
    providing reliability monitoring for LC workflows.

    Parameters
    ----------
    chain : object
        LangChain chain or agent.
    monitoring_points : list
        Points in chain to monitor.

    Examples
    --------
    >>> from langchain import OpenAI, LLMChain, PromptTemplate
    >>> llm = OpenAI()
    >>> chain = LLMChain(llm=llm, prompt=prompt)
    >>> adapter = LangchainAdapter(chain, monitoring_points=['input', 'output'])
    >>> result = adapter.run("Query", trust_check=True)
    """

    def __init__(
        self,
        chain: Any,
        name: str = "LangchainAdapter",
        monitoring_points: Optional[List[str]] = None,
        guardrails: Optional[List[Any]] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.chain = chain
        self.monitoring_points = monitoring_points or ["input", "output"]
        self.guardrails = guardrails or []

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def run(
        self, inputs: Dict[str, Any], trust_check: bool = True, return_trust_report: bool = False
    ) -> Dict[str, Any]:
        """
        Run chain with monitoring.

        Parameters
        ----------
        inputs : dict
            Chain inputs.
        trust_check : bool
            Whether to run trust checks.
        return_trust_report : bool
            Whether to include trust report in output.

        Returns
        -------
        dict with chain output and optional trust report.
        """
        trust_reports = []

        # Check input if requested
        if trust_check and "input" in self.monitoring_points:
            input_str = str(inputs)
            for guardrail in self.guardrails:
                report = guardrail.validate(input_str)
                trust_reports.append(report)
                if not report["is_valid"] and report["action"] == "block":
                    return {
                        "output": None,
                        "error": "Input blocked by guardrail",
                        "trust_reports": trust_reports if return_trust_report else None,
                    }

        # Run chain
        try:
            if hasattr(self.chain, "run"):
                output = self.chain.run(inputs)
            elif hasattr(self.chain, "__call__"):
                output = self.chain(inputs)
            else:
                raise ValueError("Chain has no run or call method")
        except Exception as e:
            return {
                "output": None,
                "error": str(e),
                "trust_reports": trust_reports if return_trust_report else None,
            }

        # Check output if requested
        if trust_check and "output" in self.monitoring_points:
            output_str = str(output)
            for guardrail in self.guardrails:
                report = guardrail.validate(output_str)
                trust_reports.append(report)

        result = {"output": output}
        if return_trust_report:
            result["trust_reports"] = trust_reports

        return result

    def batch_run(
        self, inputs_list: List[Dict[str, Any]], trust_check: bool = True
    ) -> List[Dict[str, Any]]:
        """Run chain on batch of inputs."""
        return [
            self.run(inputs, trust_check=trust_check, return_trust_report=True)
            for inputs in inputs_list
        ]

    def add_guardrail(self, guardrail: Any):
        """Add a guardrail to the adapter."""
        self.guardrails.append(guardrail)

    def get_chain(self) -> Any:
        """Return underlying LangChain chain."""
        return self.chain
