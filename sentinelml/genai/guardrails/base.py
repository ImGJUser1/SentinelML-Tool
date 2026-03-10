import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/genai/guardrails/base.py
"""
Base class for LLM guardrails.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from sentinelml.core.base import BaseGuardrail


class BaseLLMGuardrail(BaseGuardrail):
    """
    Base class for LLM guardrails.

    Guardrails validate and potentially modify
    LLM inputs and outputs.

    Parameters
    ----------
    fail_mode : str, default='filter'
        Action on validation failure.
    """

    VALID_FAIL_MODES = ["filter", "flag", "block", "sanitize"]

    def __init__(
        self, name: Optional[str] = None, fail_mode: str = "filter", verbose: bool = False
    ):
        super().__init__(name=name, fail_mode=fail_mode, verbose=verbose)

    @abstractmethod
    def validate(self, content: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate content.

        Parameters
        ----------
        content : str
            Text to validate.
        context : dict, optional
            Additional context.

        Returns
        -------
        dict with validation results.
        """
        pass
