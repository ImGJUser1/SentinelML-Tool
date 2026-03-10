import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/infrastructure/streaming/kafka_consumer.py
"""
Kafka consumer for real-time drift detection.
"""

import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from sentinelml.core.base import BaseSentinelComponent


class KafkaConsumer(BaseSentinelComponent):
    """
    Consume from Kafka for real-time monitoring.

    Processes streaming data through SentinelML
    components for real-time drift and anomaly detection.

    Parameters
    ----------
    bootstrap_servers : str
        Kafka bootstrap servers.
    topic : str
        Topic to consume.
    group_id : str
        Consumer group ID.
    detector : object
        SentinelML detector to apply.

    Examples
    --------
    >>> consumer = KafkaConsumer(
    ...     bootstrap_servers='localhost:9092',
    ...     topic='ml-predictions',
    ...     detector=MMDDriftDetector()
    ... )
    >>> consumer.start(callback=alert_function)
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        detector: Any,
        name: str = "KafkaConsumer",
        auto_offset_reset: str = "latest",
        batch_size: int = 100,
        verbose: bool = False,
    ):
        super().__init__(name=name, verbose=verbose)
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.detector = detector
        self.auto_offset_reset = auto_offset_reset
        self.batch_size = batch_size

        self._consumer = None
        self._running = False
        self._thread = None

    def fit(self, X=None, y=None):
        """Initialize Kafka consumer."""
        try:
            from kafka import KafkaConsumer as KafkaConsumerLib
        except ImportError:
            raise ImportError("kafka-python required")

        self._consumer = KafkaConsumerLib(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset=self.auto_offset_reset,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

        self.is_fitted_ = True
        return self

    def start(self, callback: Optional[Callable] = None):
        """
        Start consuming messages.

        Parameters
        ----------
        callback : callable, optional
            Function called with (message, detection_result).
        """
        if not self.is_fitted_:
            raise RuntimeError("Consumer not fitted. Call fit() first.")

        self._running = True
        self._thread = threading.Thread(target=self._consume, args=(callback,))
        self._thread.start()

        if self.verbose:
            print(f"Started consuming from {self.topic}")

    def _consume(self, callback: Optional[Callable]):
        """Main consumption loop."""
        batch = []

        for message in self._consumer:
            if not self._running:
                break

            try:
                data = message.value
                batch.append(data)

                if len(batch) >= self.batch_size:
                    self._process_batch(batch, callback)
                    batch = []

            except Exception as e:
                if self.verbose:
                    print(f"Error processing message: {e}")

        # Process remaining batch
        if batch:
            self._process_batch(batch, callback)

    def _process_batch(self, batch: List[Dict], callback: Optional[Callable]):
        """Process a batch of messages."""
        try:
            # Extract features (assume 'features' key or use whole message)
            features = []
            for item in batch:
                if "features" in item:
                    features.append(item["features"])
                else:
                    # Use numeric fields
                    feats = [v for v in item.values() if isinstance(v, (int, float))]
                    if feats:
                        features.append(feats)

            if not features:
                return

            import numpy as np

            X = np.array(features)

            # Run detection
            if hasattr(self.detector, "detect"):
                is_drift, scores = self.detector.detect(X)
            elif hasattr(self.detector, "score"):
                scores = self.detector.score(X)
                is_drift = scores < 0.5
            else:
                return

            # Call callback if provided
            if callback:
                for i, item in enumerate(batch):
                    result = {
                        "is_drift": bool(is_drift[i])
                        if hasattr(is_drift, "__getitem__")
                        else bool(is_drift),
                        "score": float(scores[i])
                        if hasattr(scores, "__getitem__")
                        else float(scores),
                        "timestamp": item.get("timestamp"),
                        "id": item.get("id"),
                    }
                    callback(item, result)

        except Exception as e:
            if self.verbose:
                print(f"Batch processing error: {e}")

    def stop(self):
        """Stop consuming."""
        self._running = False
        if self._consumer:
            self._consumer.close()
        if self._thread:
            self._thread.join(timeout=5)

        if self.verbose:
            print("Consumer stopped")

    def is_running(self) -> bool:
        """Check if consumer is running."""
        return self._running and self._thread is not None and self._thread.is_alive()
