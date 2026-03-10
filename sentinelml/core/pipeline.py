import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/core/pipeline.py (continued)
"""
Scikit-learn compatible pipeline for SentinelML components.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline as SklearnPipeline

from sentinelml.core.base import BaseDetector, BaseGuardrail, BaseSentinelComponent, BaseTrustModel
from sentinelml.core.report import DriftReport, GuardrailReport, TrustReport


class SentinelPipeline(BaseEstimator):
    """
    Pipeline for composing SentinelML components.

    Similar to sklearn.pipeline.Pipeline but designed for
    trust assessment workflows with rich reporting.

    Parameters
    ----------
    steps : list of tuples
        List of (name, component) pairs.
    memory : str or object, optional
        Used to cache the fitted transformers.
    verbose : bool, default=False
        If True, print progress messages.

    Examples
    --------
    >>> from sentinelml.traditional import MMDDriftDetector, MahalanobisTrust
    >>> from sentinelml.deep_learning import MCDropoutUncertainty

    >>> pipeline = SentinelPipeline([
    ...     ('drift', MMDDriftDetector(threshold=0.05)),
    ...     ('trust', MahalanobisTrust()),
    ...     ('uncertainty', MCDropoutUncertainty(model, n_samples=50))
    ... ])

    >>> pipeline.fit(X_ref)
    >>> report = pipeline.assess(X_test)
    """

    def __init__(
        self,
        steps: List[Tuple[str, BaseSentinelComponent]],
        memory: Optional[Any] = None,
        verbose: bool = False,
    ):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        self.named_steps = dict(steps)
        self._validate_steps()

    def _validate_steps(self):
        """Validate pipeline steps."""
        names = set()
        for name, component in self.steps:
            if not isinstance(name, str):
                raise TypeError(f"Step name must be string, got {type(name)}")
            if name in names:
                raise ValueError(f"Duplicate step name: {name}")
            names.add(name)

            if not isinstance(component, BaseSentinelComponent):
                raise TypeError(
                    f"Step {name} must be BaseSentinelComponent, " f"got {type(component)}"
                )

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None, **fit_params
    ) -> "SentinelPipeline":
        """
        Fit all steps in the pipeline.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Target values.
        **fit_params : dict
            Parameters passed to fit methods.

        Returns
        -------
        self : SentinelPipeline
            Fitted pipeline.
        """
        X = np.asarray(X)

        for name, component in self.steps:
            if self.verbose:
                print(f"[SentinelPipeline] Fitting {name}...")

            # Pass appropriate parameters based on component type
            if isinstance(component, BaseTrustModel):
                component.fit(X, y)
            else:
                component.fit(X, y)

        return self

    def assess(self, X: npt.ArrayLike, sample_id: Optional[str] = None) -> TrustReport:
        """
        Comprehensive assessment through all pipeline steps.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to assess.
        sample_id : str, optional
            Identifier for this assessment.

        Returns
        -------
        report : TrustReport
            Comprehensive trust assessment.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Initialize report components
        drift_report = None
        familiarity_score = None
        uncertainty_score = None
        guardrail_reports = []
        raw_scores = {}

        # Run through each step
        for name, component in self.steps:
            if isinstance(component, BaseDetector):
                is_drift, drift_scores = component.detect(X)
                drift_report = DriftReport(
                    component_name=name,
                    drift_detected=bool(is_drift[0]) if len(is_drift) > 0 else False,
                    drift_score=float(np.mean(drift_scores)),
                    metadata={"scores": drift_scores.tolist()},
                )
                raw_scores[f"{name}_drift"] = float(np.mean(drift_scores))

            elif isinstance(component, BaseTrustModel):
                scores = component.score(X)
                if "familiarity" in name.lower():
                    familiarity_score = float(np.mean(scores))
                elif "uncertainty" in name.lower():
                    uncertainty_score = float(np.mean(scores))
                else:
                    uncertainty_score = float(np.mean(scores))
                raw_scores[name] = float(np.mean(scores))

            elif isinstance(component, BaseGuardrail):
                for i, x in enumerate(X):
                    result = component.validate(x)
                    guardrail_reports.append(
                        GuardrailReport(
                            component_name=name,
                            is_valid=result["is_valid"],
                            validation_score=result.get("score", 0.0),
                            action_taken=result.get("action", "pass"),
                            metadata=result.get("metadata", {}),
                        )
                    )

        # Compute aggregate trust score
        trust_components = []
        if familiarity_score is not None:
            trust_components.append(familiarity_score)
        if uncertainty_score is not None:
            trust_components.append(uncertainty_score)

        trust_score = np.mean(trust_components) if trust_components else 0.5

        # Adjust for drift and violations
        if drift_report and drift_report.drift_detected:
            trust_score *= 0.5  # Penalize drift

        for gr in guardrail_reports:
            if not gr.is_valid:
                trust_score *= 0.8  # Penalize violations

        return TrustReport(
            trust_score=float(np.clip(trust_score, 0, 1)),
            confidence=float(1.0 - (drift_report.drift_score if drift_report else 0)),
            drift_report=drift_report,
            familiarity_score=familiarity_score,
            uncertainty_score=uncertainty_score,
            guardrail_reports=guardrail_reports,
            sample_id=sample_id,
            raw_scores=raw_scores,
        )

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Predict trust probabilities (sklearn compatibility).

        Returns
        -------
        probabilities : ndarray of shape (n_samples, 2)
            [1-trust, trust] probabilities.
        """
        report = self.assess(X)
        trust = report.trust_score
        return np.array([[1 - trust, trust]])

    def score(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> float:
        """
        Compute mean trust score (sklearn compatibility).
        """
        report = self.assess(X)
        return report.trust_score

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = {"steps": self.steps, "memory": self.memory, "verbose": self.verbose}
        if deep:
            for name, component in self.steps:
                for key, value in component.get_params(deep=True).items():
                    params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params) -> "SentinelPipeline":
        """Set parameters."""
        if "memory" in params:
            self.memory = params.pop("memory")
        if "verbose" in params:
            self.verbose = params.pop("verbose")

        # Handle step parameters
        step_params = {}
        for key, value in params.items():
            if "__" in key:
                step_name, param_name = key.split("__", 1)
                if step_name not in step_params:
                    step_params[step_name] = {}
                step_params[step_name][param_name] = value

        for step_name, step_p in step_params.items():
            if step_name in self.named_steps:
                self.named_steps[step_name].set_params(**step_p)

        return self
