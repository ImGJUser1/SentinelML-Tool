# sentinelml/core/report.py
"""
Reporting classes for SentinelML assessments.

Provides structured, serializable reports for trust scores,
drift detection, and guardrail validations.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt


@dataclass
class ComponentReport:
    """Base report from a single component."""

    component_name: str
    component_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class DriftReport(ComponentReport):
    """Report from drift detection."""

    drift_detected: bool = False
    p_value: float = 1.0
    drift_score: float = 0.0
    affected_features: List[str] = field(default_factory=list)
    feature_scores: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.component_type = "drift_detector"

    @property
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if drift is statistically significant."""
        return self.p_value < alpha


@dataclass
class GuardrailReport(ComponentReport):
    """Report from guardrail validation."""

    is_valid: bool = True
    validation_score: float = 1.0
    violations: List[Dict[str, Any]] = field(default_factory=list)
    action_taken: str = "pass"  # 'pass', 'filter', 'block', 'sanitize'
    sanitized_content: Optional[Any] = None

    def __post_init__(self):
        self.component_type = "guardrail"


@dataclass
class TrustReport:
    """
    Comprehensive trust assessment report.

    Aggregates results from multiple components into a unified view.
    """

    # Core metrics
    trust_score: float = 0.0
    confidence: float = 0.0

    # Component reports
    drift_report: Optional[DriftReport] = None
    familiarity_score: Optional[float] = None
    uncertainty_score: Optional[float] = None
    guardrail_reports: List[GuardrailReport] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    sample_id: Optional[str] = None
    model_version: Optional[str] = None

    # Raw scores for debugging
    raw_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate score ranges."""
        if not 0 <= self.trust_score <= 1:
            raise ValueError(f"trust_score must be in [0, 1], got {self.trust_score}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def is_trustworthy(self, threshold: float = 0.7) -> bool:
        """Check if sample meets trust threshold."""
        return self.trust_score >= threshold

    @property
    def has_drift(self) -> bool:
        """Check if drift was detected."""
        return self.drift_report is not None and self.drift_report.drift_detected

    @property
    def has_violations(self) -> bool:
        """Check if any guardrail violations occurred."""
        return any(not r.is_valid for r in self.guardrail_reports)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trust_score": self.trust_score,
            "confidence": self.confidence,
            "is_trustworthy": self.is_trustworthy,
            "has_drift": self.has_drift,
            "has_violations": self.has_violations,
            "drift_report": self.drift_report.to_dict() if self.drift_report else None,
            "familiarity_score": self.familiarity_score,
            "uncertainty_score": self.uncertainty_score,
            "guardrail_reports": [r.to_dict() for r in self.guardrail_reports],
            "timestamp": self.timestamp.isoformat(),
            "sample_id": self.sample_id,
            "model_version": self.model_version,
            "raw_scores": self.raw_scores,
            "metadata": self.metadata,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Trust Report (ID: {self.sample_id or 'N/A'})",
            f"  Trust Score: {self.trust_score:.3f} (threshold: 0.7)",
            f"  Confidence:  {self.confidence:.3f}",
            f"  Status:      {'TRUSTWORTHY' if self.is_trustworthy else 'UNTRUSTWORTHY'}",
        ]

        if self.has_drift:
            lines.append(f"  ⚠️  DRIFT DETECTED (p={self.drift_report.p_value:.4f})")

        if self.has_violations:
            lines.append(
                f"  🚫 GUARDRAIL VIOLATIONS: {len([r for r in self.guardrail_reports if not r.is_valid])}"
            )

        return "\n".join(lines)
