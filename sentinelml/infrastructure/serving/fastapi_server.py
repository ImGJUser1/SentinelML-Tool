import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/infrastructure/serving/fastapi_server.py
"""
FastAPI server for SentinelML services.
"""

import json
from typing import Any, Callable, Dict, List, Optional

from sentinelml.core.base import BaseSentinelComponent


class FastAPIServer(BaseSentinelComponent):
    """
    FastAPI server for serving SentinelML components.

    Provides REST API for drift detection, trust scoring,
    and guardrail validation.

    Parameters
    ----------
    sentinel : object
        Sentinel instance to serve.
    host : str, default='0.0.0.0'
        Server host.
    port : int, default=8000
        Server port.

    Examples
    --------
    >>> from sentinelml import Sentinel
    >>> sentinel = Sentinel(drift_detector=detector, trust_model=trust)
    >>> server = FastAPIServer(sentinel, port=8000)
    >>> server.start()
    """

    def __init__(
        self,
        sentinel: Any,
        name: str = "FastAPIServer",
        host: str = "0.0.0.0",
        port: int = 8000,
        log_level: str = "info",
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.sentinel = sentinel
        self.host = host
        self.port = port
        self.log_level = log_level

        self._app = None
        self._server = None

    def fit(self, X=None, y=None):
        """Initialize FastAPI app."""
        try:
            import uvicorn
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
        except ImportError:
            raise ImportError("fastapi and uvicorn required")

        self._app = FastAPI(title="SentinelML API", version="2.0.0")

        # Define request/response models
        class AssessRequest(BaseModel):
            data: List[List[float]]
            sample_id: Optional[str] = None
            context: Optional[Dict] = None

        class AssessResponse(BaseModel):
            trust_score: float
            confidence: float
            is_trustworthy: bool
            drift_detected: bool
            has_violations: bool
            raw_scores: Dict[str, float]

        # Define endpoints
        @self._app.post("/assess", response_model=AssessResponse)
        async def assess(request: AssessRequest):
            try:
                import numpy as np

                X = np.array(request.data)

                report = self.sentinel.assess(
                    X, sample_id=request.sample_id, context=request.context
                )

                return {
                    "trust_score": report.trust_score,
                    "confidence": report.confidence,
                    "is_trustworthy": report.is_trustworthy,
                    "drift_detected": report.has_drift,
                    "has_violations": report.has_violations,
                    "raw_scores": report.raw_scores,
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self._app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "sentinel_fitted": getattr(self.sentinel, "is_fitted_", False),
            }

        @self._app.get("/summary")
        async def summary():
            if hasattr(self.sentinel, "summary"):
                return self.sentinel.summary()
            return {"message": "No summary available"}

        self.is_fitted_ = True
        return self

    def start(self, blocking: bool = False):
        """
        Start server.

        Parameters
        ----------
        blocking : bool
            If True, blocks until server stops.
        """
        if not self.is_fitted_:
            raise RuntimeError("Server not fitted. Call fit() first.")

        import uvicorn

        config = uvicorn.Config(self._app, host=self.host, port=self.port, log_level=self.log_level)

        self._server = uvicorn.Server(config)

        if blocking:
            self._server.run()
        else:
            import threading

            self._thread = threading.Thread(target=self._server.run)
            self._thread.start()

            if self.verbose:
                print(f"Server started at http://{self.host}:{self.port}")

    def stop(self):
        """Stop server."""
        if self._server:
            self._server.should_exit = True
        if hasattr(self, "_thread"):
            self._thread.join(timeout=5)

    def get_app(self):
        """Return FastAPI app (for testing/mounting)."""
        return self._app
