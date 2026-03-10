import re
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/rag/end_to_end/latency_tracker.py
"""
Performance and latency tracking for RAG pipelines.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from sentinelml.core.base import BaseSentinelComponent


@dataclass
class LatencyRecord:
    """Record of a single operation's latency."""

    operation: str
    start_time: float
    end_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class LatencyTracker(BaseSentinelComponent):
    """
    Track latency of RAG pipeline components.

    Measures retrieval time, generation time, and
    end-to-end latency for performance monitoring.

    Parameters
    ----------
    track_components : list, default=['retrieval', 'generation']
        Which components to track.

    Examples
    --------
    >>> tracker = LatencyTracker()
    >>> with tracker.track('retrieval'):
    ...     docs = retriever.retrieve(query)
    >>> report = tracker.get_report()
    """

    def __init__(
        self,
        name: str = "LatencyTracker",
        track_components: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.track_components = track_components or ["retrieval", "generation", "total"]
        self.records: List[LatencyRecord] = []
        self._current_operations: Dict[str, float] = {}

    def fit(self, X=None, y=None):
        """No fitting required."""
        self.is_fitted_ = True
        return self

    def track(self, operation: str, metadata: Optional[Dict] = None):
        """
        Context manager for tracking an operation.

        Usage:
            with tracker.track('retrieval'):
                docs = retriever.retrieve(query)
        """
        return _LatencyContext(self, operation, metadata or {})

    def start(self, operation: str, metadata: Optional[Dict] = None):
        """Start tracking an operation."""
        self._current_operations[operation] = time.time()

    def end(self, operation: str, metadata: Optional[Dict] = None):
        """End tracking an operation."""
        if operation not in self._current_operations:
            raise ValueError(f"Operation {operation} not started")

        start_time = self._current_operations.pop(operation)
        end_time = time.time()

        record = LatencyRecord(
            operation=operation, start_time=start_time, end_time=end_time, metadata=metadata or {}
        )
        self.records.append(record)

        if self.verbose:
            print(f"{operation}: {record.duration_ms:.2f}ms")

        return record.duration_ms

    def get_report(self) -> Dict[str, Any]:
        """
        Generate latency report.

        Returns
        -------
        dict with latency statistics.
        """
        if not self.records:
            return {"message": "No records available"}

        # Group by operation
        by_operation = defaultdict(list)
        for record in self.records:
            by_operation[record.operation].append(record.duration_ms)

        report = {}
        for op, durations in by_operation.items():
            report[op] = {
                "count": len(durations),
                "mean_ms": float(np.mean(durations)),
                "median_ms": float(np.median(durations)),
                "std_ms": float(np.std(durations)),
                "min_ms": float(np.min(durations)),
                "max_ms": float(np.max(durations)),
                "p95_ms": float(np.percentile(durations, 95)),
                "p99_ms": float(np.percentile(durations, 99)),
            }

        # Overall statistics
        all_durations = [r.duration_ms for r in self.records]
        report["overall"] = {
            "total_operations": len(self.records),
            "total_time_ms": sum(all_durations),
            "mean_operation_ms": float(np.mean(all_durations)),
        }

        return report

    def reset(self):
        """Clear all records."""
        self.records = []
        self._current_operations = {}


class _LatencyContext:
    """Context manager for latency tracking."""

    def __init__(self, tracker: LatencyTracker, operation: str, metadata: Dict):
        self.tracker = tracker
        self.operation = operation
        self.metadata = metadata

    def __enter__(self):
        self.tracker.start(self.operation, self.metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.end(
            self.operation, {**self.metadata, "error": str(exc_val) if exc_val else None}
        )
        return False  # Don't suppress exceptions
