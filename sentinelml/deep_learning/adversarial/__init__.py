# sentinelml/deep_learning/adversarial/__init__.py
"""Adversarial robustness detection."""

from sentinelml.deep_learning.adversarial.boundary_attack import PGDDetector
from sentinelml.deep_learning.adversarial.fgsm_detector import FGMDetector

__all__ = [
    "FGMDetector",
    "PGDDetector",
]
