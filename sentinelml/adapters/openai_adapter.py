import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/adapters/openai_adapter.py
"""
OpenAI API adapter with monitoring.
"""

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from sentinelml.core.base import BaseSentinelComponent


class OpenAIAdapter(BaseSentinelComponent):
    """
    Adapter for OpenAI API with built-in monitoring.

    Wraps OpenAI client with trust scoring, rate limiting,
    and response validation.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key (or use env var).
    model : str, default='gpt-4'
        Model to use.
    monitoring : bool, default=True
        Enable response monitoring.

    Examples
    --------
    >>> adapter = OpenAIAdapter(model='gpt-4')
    >>> response = adapter.complete("Prompt", trust_check=True)
    >>> embedding = adapter.embed("Text")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        name: str = "OpenAIAdapter",
        model: str = "gpt-4",
        monitoring: bool = True,
        guardrails: Optional[List[Any]] = None,
        rate_limit_per_minute: int = 60,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.api_key = api_key
        self.model = model
        self.monitoring = monitoring
        self.guardrails = guardrails or []
        self.rate_limit = rate_limit_per_minute

        self._client = None
        self._call_times: List[float] = []

    def fit(self, X=None, y=None):
        """Initialize OpenAI client."""
        try:
            import openai

            if self.api_key:
                openai.api_key = self.api_key
            self._client = openai
        except ImportError:
            raise ImportError("openai package required")

        self.is_fitted_ = True
        return self

    def _check_rate_limit(self):
        """Check and enforce rate limit."""
        now = time.time()
        minute_ago = now - 60

        # Remove old calls
        self._call_times = [t for t in self._call_times if t > minute_ago]

        if len(self._call_times) >= self.rate_limit:
            sleep_time = 60 - (now - self._call_times[0])
            if sleep_time > 0:
                if self.verbose:
                    print(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

        self._call_times.append(now)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
        trust_check: bool = True,
        return_metadata: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate completion with monitoring.

        Parameters
        ----------
        prompt : str
            Input prompt.
        max_tokens : int
            Maximum tokens to generate.
        temperature : float
            Sampling temperature.
        trust_check : bool
            Whether to validate response.

        Returns
        -------
        dict with completion and trust information.
        """
        self._check_rate_limit()

        # Check input guardrails
        if trust_check:
            for guardrail in self.guardrails:
                result = guardrail.validate(prompt)
                if not result["is_valid"] and result["action"] == "block":
                    return {
                        "completion": None,
                        "error": "Prompt blocked by guardrail",
                        "guardrail_result": result,
                    }

        # Call API
        try:
            response = self._client.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            completion = response.choices[0].message.content

        except Exception as e:
            return {"completion": None, "error": str(e)}

        # Check output guardrails
        guardrail_results = []
        if trust_check:
            for guardrail in self.guardrails:
                result = guardrail.validate(
                    completion, context={"prompt": prompt, "model": self.model}
                )
                guardrail_results.append(result)

        result = {
            "completion": completion,
            "guardrail_results": guardrail_results if trust_check else None,
        }

        if return_metadata:
            result["metadata"] = {
                "model": response.model,
                "usage": dict(response.usage),
                "finish_reason": response.choices[0].finish_reason,
            }

        return result

    def embed(self, texts: List[str], model: str = "text-embedding-ada-002") -> np.ndarray:
        """
        Get embeddings for texts.

        Parameters
        ----------
        texts : list of str
            Texts to embed.
        model : str
            Embedding model.

        Returns
        -------
        embeddings : ndarray
            Text embeddings.
        """
        self._check_rate_limit()

        # Process in batches
        all_embeddings = []

        for i in range(0, len(texts), 100):  # OpenAI batch limit
            batch = texts[i : i + 100]

            try:
                response = self._client.Embedding.create(input=batch, model=model)

                batch_embeddings = [item["embedding"] for item in response["data"]]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                if self.verbose:
                    print(f"Embedding error: {e}")
                # Return zeros for failed batch
                all_embeddings.extend([[0.0] * 1536] * len(batch))

        return np.array(all_embeddings)

    def get_logprobs(self, text: str) -> List[float]:
        """
        Get token log probabilities (requires logprobs=1 in API call).

        Note: This requires a separate completion call with logprobs enabled.
        """
        self._check_rate_limit()

        try:
            response = self._client.Completion.create(
                model=self.model, prompt=text, max_tokens=0, echo=True, logprobs=1
            )

            # Extract logprobs
            logprobs_data = response["choices"][0].get("logprobs", {})
            token_logprobs = logprobs_data.get("token_logprobs", [])

            return token_logprobs

        except Exception as e:
            if self.verbose:
                print(f"Logprobs error: {e}")
            return []

    def add_guardrail(self, guardrail: Any):
        """Add guardrail."""
        self.guardrails.append(guardrail)
