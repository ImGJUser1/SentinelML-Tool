import time
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/core/sentinel.py
"""
Main Sentinel orchestrator - the unified entry point for all reliability assessments.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseDetector, BaseGuardrail, BaseSentinelComponent, BaseTrustModel
from sentinelml.core.ensemble import AdaptiveTrustEnsemble
from sentinelml.core.report import DriftReport, TrustReport

logger = logging.getLogger(__name__)


class Sentinel:
    """
    Unified reliability orchestrator for AI/ML systems.

    The main entry point that coordinates drift detection, trust modeling,
    guardrails, and uncertainty quantification across all AI modalities.

    Parameters
    ----------
    drift_detector : BaseDetector, optional
        Component for detecting distribution shift.
    trust_model : BaseTrustModel, optional
        Primary trust scoring model.
    guardrails : list of BaseGuardrail, optional
        Input/output validation rules.
    ensemble_validators : list of BaseTrustModel, optional
        Multiple validators for adaptive ensemble.
    adaptive_weights : bool, default=True
        Whether to learn optimal validator weights.
    verbose : bool, default=False
        Enable verbose logging.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the sentinel has been fitted.
    reference_data_ : ndarray
        Reference distribution data.

    Examples
    --------
    >>> from sentinelml import Sentinel
    >>> from sentinelml.traditional import MMDDriftDetector, MahalanobisTrust

    >>> sentinel = Sentinel(
    ...     drift_detector=MMDDriftDetector(),
    ...     trust_model=MahalanobisTrust()
    ... )

    >>> sentinel.fit(X_reference)
    >>> report = sentinel.assess(X_new)
    >>> print(report.summary())
    """

    def __init__(
        self,
        drift_detector: Optional[BaseDetector] = None,
        trust_model: Optional[BaseTrustModel] = None,
        guardrails: Optional[List[BaseGuardrail]] = None,
        ensemble_validators: Optional[List[BaseTrustModel]] = None,
        adaptive_weights: bool = True,
        verbose: bool = False,
    ):
        self.drift_detector = drift_detector
        self.trust_model = trust_model
        self.guardrails = guardrails or []
        self.ensemble_validators = ensemble_validators
        self.adaptive_weights = adaptive_weights
        self.verbose = verbose

        self.is_fitted_ = False
        self.reference_data_: Optional[npt.NDArray] = None
        self.ensemble_: Optional[AdaptiveTrustEnsemble] = None
        self.history_: List[TrustReport] = []

        # Configuration
        self.drift_penalty_ = 0.5
        self.violation_penalty_ = 0.8

    def fit(
        self,
        X: npt.ArrayLike,
        y: Optional[npt.NDArray] = None,
        predictions: Optional[npt.NDArray] = None,
        true_labels: Optional[npt.NDArray] = None,
    ) -> "Sentinel":
        """
        Fit all components to reference data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Reference data representing expected distribution.
        y : array-like, optional
            Target values for supervised components.
        predictions : array-like, optional
            Model predictions on X (for calibration).
        true_labels : array-like, optional
            Ground truth for weight optimization.

        Returns
        -------
        self : Sentinel
            Fitted sentinel.
        """
        X = np.asarray(X)
        self.reference_data_ = X.copy()

        if self.verbose:
            logger.info(f"Fitting Sentinel on {len(X)} samples...")

        # Fit drift detector
        if self.drift_detector is not None:
            if self.verbose:
                logger.info("Fitting drift detector...")
            self.drift_detector.fit(X, y)

        # Fit trust model
        if self.trust_model is not None:
            if self.verbose:
                logger.info("Fitting trust model...")
            self.trust_model.fit(X, y)

            # Calibrate if labels available
            if predictions is not None and true_labels is not None:
                self.trust_model.calibrate(X, true_labels, predictions)

        # Fit guardrails
        for guardrail in self.guardrails:
            if self.verbose:
                logger.info(f"Fitting guardrail: {guardrail.name}")
            guardrail.fit(X, y)

        # Setup ensemble if multiple validators provided
        if self.ensemble_validators:
            validators = self.ensemble_validators
            if self.trust_model:
                validators = [self.trust_model] + validators

            self.ensemble_ = AdaptiveTrustEnsemble(validators=validators, verbose=self.verbose)
            self.ensemble_.fit(X, y, predictions, true_labels)
        elif self.trust_model:
            # Single trust model
            self.ensemble_ = AdaptiveTrustEnsemble(
                validators=[self.trust_model], verbose=self.verbose
            )
            self.ensemble_.fit(X, y, predictions, true_labels)

        self.is_fitted_ = True

        if self.verbose:
            logger.info("Sentinel fitting complete.")

        return self

    def assess(
        self,
        X: npt.ArrayLike,
        sample_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TrustReport:
        """
        Comprehensive reliability assessment.

        Parameters
        ----------
        X : array-like
            Data to assess. Can be single sample or batch.
        sample_id : str, optional
            Identifier for tracking.
        context : dict, optional
            Additional context (model version, environment, etc.).

        Returns
        -------
        TrustReport
            Complete assessment with trust scores, drift status, and violations.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Initialize report components
        drift_report = None
        guardrail_reports = []
        raw_scores = {}

        # 1. Drift Detection
        if self.drift_detector is not None:
            is_drift, drift_scores = self.drift_detector.detect(X)
            drift_report = DriftReport(
                component_name=self.drift_detector.name,
                drift_detected=bool(is_drift.any()),
                drift_score=float(np.mean(drift_scores)),
                p_value=float(np.min(drift_scores)) if len(drift_scores) > 0 else 1.0,
                metadata={"per_sample_scores": drift_scores.tolist()},
            )
            raw_scores["drift"] = float(np.mean(drift_scores))

        # 2. Trust Scoring (via ensemble)
        if self.ensemble_ is not None:
            trust_scores = self.ensemble_.score(X)
            base_trust = float(np.mean(trust_scores))
            raw_scores["base_trust"] = base_trust
        else:
            base_trust = 0.5
            logger.warning("No trust model configured, using default trust=0.5")

        # 3. Guardrail Validation
        for guardrail in self.guardrails:
            for i, x in enumerate(X):
                try:
                    result = guardrail.validate(x, context)
                    guardrail_reports.append(
                        GuardrailReport(
                            component_name=guardrail.name,
                            is_valid=result.get("is_valid", True),
                            validation_score=result.get("score", 1.0),
                            action_taken=result.get("action", "pass"),
                            violations=result.get("violations", []),
                            metadata=result.get("metadata", {}),
                        )
                    )
                    raw_scores[f"guardrail_{guardrail.name}"] = result.get("score", 1.0)
                except Exception as e:
                    logger.error(f"Guardrail {guardrail.name} failed: {e}")
                    guardrail_reports.append(
                        GuardrailReport(
                            component_name=guardrail.name,
                            is_valid=False,
                            validation_score=0.0,
                            action_taken="block",
                            metadata={"error": str(e)},
                        )
                    )

        # 4. Compute Final Trust Score
        final_trust = base_trust

        # Apply drift penalty
        if drift_report and drift_report.drift_detected:
            final_trust *= self.drift_penalty_
            if self.verbose:
                logger.warning(
                    f"Drift detected! Trust penalized: {base_trust:.3f} -> {final_trust:.3f}"
                )

        # Apply guardrail penalties
        violations = [g for g in guardrail_reports if not g.is_valid]
        for _ in violations:
            final_trust *= self.violation_penalty_

        if violations and self.verbose:
            logger.warning(f"{len(violations)} guardrail violations. Trust: {final_trust:.3f}")

        # Compute confidence based on score variance
        confidence = 1.0 - (drift_report.drift_score if drift_report else 0.0)

        # Create report
        report = TrustReport(
            trust_score=float(np.clip(final_trust, 0.0, 1.0)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            drift_report=drift_report,
            uncertainty_score=base_trust,
            guardrail_reports=guardrail_reports,
            sample_id=sample_id or f"sample_{datetime.now().isoformat()}",
            model_version=context.get("model_version") if context else None,
            raw_scores=raw_scores,
            metadata={
                "n_samples": len(X),
                "n_guardrails": len(self.guardrails),
                "n_violations": len(violations),
                "context": context or {},
            },
        )

        # Store in history
        self.history_.append(report)

        return report

    def assess_stream(self, X_stream, callback: Optional[Callable[[TrustReport], None]] = None):
        """
        Assess streaming data.

        Parameters
        ----------
        X_stream : iterable
            Stream of data samples.
        callback : callable, optional
            Function called with each report.
        """
        for i, X in enumerate(X_stream):
            report = self.assess(X, sample_id=f"stream_{i}")
            if callback:
                callback(report)
            yield report

    def update_reference(
        self, X_new: npt.ArrayLike, strategy: str = "incremental", rate: float = 0.1
    ) -> "Sentinel":
        """
        Update reference distribution with new data.

        Parameters
        ----------
        X_new : array-like
            New data to incorporate.
        strategy : str, default='incremental'
            Update strategy ('incremental', 'window', 'full').
        rate : float, default=0.1
            For incremental: fraction of new data to incorporate.
        """
        X_new = np.asarray(X_new)

        if strategy == "incremental":
            # Random replacement with exponential moving average
            n_replace = int(len(self.reference_data_) * rate)
            indices = np.random.choice(len(self.reference_data_), n_replace, replace=False)
            new_samples = X_new[
                np.random.choice(len(X_new), min(n_replace, len(X_new)), replace=False)
            ]
            self.reference_data_[indices] = new_samples

        elif strategy == "window":
            # Rolling window: drop oldest, add newest
            window_size = len(self.reference_data_)
            self.reference_data_ = np.vstack([self.reference_data_[len(X_new) :], X_new])

        elif strategy == "full":
            # Complete refit
            self.reference_data_ = np.vstack([self.reference_data_, X_new])

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Refit components
        return self.fit(self.reference_data_)

    def get_history(self, n: Optional[int] = None) -> List[TrustReport]:
        """
        Retrieve assessment history.

        Parameters
        ----------
        n : int, optional
            Number of recent reports to return (None for all).
        """
        if n is None:
            return self.history_
        return self.history_[-n:]

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics of all assessments."""
        if not self.history_:
            return {"message": "No assessments recorded"}

        trust_scores = [r.trust_score for r in self.history_]
        drift_rate = sum(1 for r in self.history_ if r.has_drift) / len(self.history_)
        violation_rate = sum(1 for r in self.history_ if r.has_violations) / len(self.history_)

        return {
            "n_assessments": len(self.history_),
            "mean_trust": np.mean(trust_scores),
            "std_trust": np.std(trust_scores),
            "min_trust": np.min(trust_scores),
            "max_trust": np.max(trust_scores),
            "drift_rate": drift_rate,
            "violation_rate": violation_rate,
            "untrustworthy_rate": sum(1 for r in self.history_ if not r.is_trustworthy)
            / len(self.history_),
        }

    def _check_is_fitted(self):
        """Validate that sentinel is fitted."""
        if not self.is_fitted_:
            raise RuntimeError("Sentinel is not fitted. Call fit() first.")

    def save(self, path: str):
        """Serialize sentinel to disk."""
        import joblib

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "Sentinel":
        """Load serialized sentinel."""
        import joblib

        return joblib.load(path)
