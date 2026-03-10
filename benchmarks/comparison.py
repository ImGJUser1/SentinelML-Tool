import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
"""
Benchmarking and comparison utilities for SentinelML v2.0.

Provides comprehensive evaluation against baseline methods for:
- Anomaly detection (Isolation Forest, LOF, One-Class SVM)
- Uncertainty estimation (Entropy, Margin, Calibration)
- Drift detection (ADWIN, Page-Hinkley, Kolmogorov-Smirnov)
- OOD detection (MSP, ODIN, Mahalanobis)
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def _check_sklearn():
    """Check if scikit-learn is available."""
    try:
        import sklearn

        return True
    except ImportError:
        raise ImportError(
            "scikit-learn required for benchmarking. Install: pip install scikit-learn"
        )


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    method_name: str
    scores: np.ndarray
    labels: Optional[np.ndarray] = None
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method_name": self.method_name,
            "scores": self.scores.tolist() if isinstance(self.scores, np.ndarray) else self.scores,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "execution_time": self.execution_time,
            "metadata": self.metadata or {},
        }


class BenchmarkComparison:
    """
    Comprehensive benchmarking suite for comparing SentinelML against baselines.

    Supports multiple evaluation scenarios:
    - Anomaly detection (unsupervised)
    - OOD detection (supervised with OOD labels)
    - Uncertainty calibration (with ground truth)
    - Drift detection (temporal)

    Parameters
    ----------
    sentinel : Sentinel
        Configured Sentinel instance
    model : Any, optional
        ML model being monitored (for prediction-based baselines)
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print progress

    Examples
    --------
    >>> from sentinelml import Sentinel, MahalanobisTrust
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> # Setup
    >>> model = RandomForestClassifier().fit(X_train, y_train)
    >>> sentinel = Sentinel(trust_model=MahalanobisTrust())
    >>> sentinel.fit(X_train)
    >>>
    >>> # Run benchmark
    >>> benchmark = BenchmarkComparison(sentinel, model=model)
    >>> results = benchmark.evaluate_anomaly_detection(X_test, contamination=0.1)
    >>> benchmark.plot_comparison(results)
    """

    def __init__(
        self, sentinel, model: Optional[Any] = None, random_state: int = 42, verbose: bool = True
    ):
        _check_sklearn()
        self.sentinel = sentinel
        self.model = model
        self.random_state = random_state
        self.verbose = verbose
        self.results_history: List[Dict[str, BenchmarkResult]] = []

    def _log(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(f"[Benchmark] {msg}")

    def evaluate_anomaly_detection(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
        contamination: float = 0.1,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare anomaly detection performance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
        labels : array-like of shape (n_samples,), optional
            Ground truth labels (1 for anomaly, 0 for normal)
        contamination : float
            Expected proportion of anomalies (for unsupervised methods)
        methods : list of str, optional
            Methods to compare. Default: ['sentinel', 'isolation_forest', 'lof', 'one_class_svm']

        Returns
        -------
        dict of BenchmarkResult
            Results for each method
        """
        import time

        methods = methods or ["sentinel", "isolation_forest", "lof", "one_class_svm"]
        X = np.asarray(X)
        results = {}

        self._log(f"Running anomaly detection benchmark on {len(X)} samples...")

        # SentinelML
        if "sentinel" in methods:
            self._log("Evaluating SentinelML...")
            start = time.time()

            sentinel_scores = []
            for x in X:
                result = self.sentinel.assess(x)
                # Convert trust score to anomaly score (1 - trust)
                sentinel_scores.append(1.0 - result.trust_score)

            sentinel_scores = np.array(sentinel_scores)
            exec_time = time.time() - start

            result = BenchmarkResult(
                method_name="SentinelML",
                scores=sentinel_scores,
                labels=labels,
                execution_time=exec_time,
                metadata={"contamination": contamination},
            )

            if labels is not None:
                result.auc_roc = roc_auc_score(labels, sentinel_scores)
                result.auc_pr = average_precision_score(labels, sentinel_scores)
                self._log(f"  AUC-ROC: {result.auc_roc:.3f}, AUC-PR: {result.auc_pr:.3f}")

            results["sentinel"] = result

        # Isolation Forest
        if "isolation_forest" in methods:
            self._log("Evaluating Isolation Forest...")
            start = time.time()

            iso = IsolationForest(
                contamination=contamination, random_state=self.random_state, n_estimators=100
            )
            iso.fit(X)
            # Negative scores (higher = more anomalous)
            iso_scores = -iso.score_samples(X)
            exec_time = time.time() - start

            result = BenchmarkResult(
                method_name="Isolation Forest",
                scores=iso_scores,
                labels=labels,
                execution_time=exec_time,
            )

            if labels is not None:
                result.auc_roc = roc_auc_score(labels, iso_scores)
                result.auc_pr = average_precision_score(labels, iso_scores)
                self._log(f"  AUC-ROC: {result.auc_roc:.3f}, AUC-PR: {result.auc_pr:.3f}")

            results["isolation_forest"] = result

        # Local Outlier Factor
        if "lof" in methods:
            self._log("Evaluating Local Outlier Factor...")
            start = time.time()

            lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
            lof.fit(X)
            # Negative scores (higher = more anomalous)
            lof_scores = -lof.score_samples(X)
            exec_time = time.time() - start

            result = BenchmarkResult(
                method_name="Local Outlier Factor",
                scores=lof_scores,
                labels=labels,
                execution_time=exec_time,
            )

            if labels is not None:
                result.auc_roc = roc_auc_score(labels, lof_scores)
                result.auc_pr = average_precision_score(labels, lof_scores)
                self._log(f"  AUC-ROC: {result.auc_roc:.3f}, AUC-PR: {result.auc_pr:.3f}")

            results["lof"] = result

        # One-Class SVM
        if "one_class_svm" in methods:
            self._log("Evaluating One-Class SVM...")
            start = time.time()

            # Subsample for large datasets
            if len(X) > 10000:
                idx = np.random.choice(len(X), 10000, replace=False)
                X_train_svm = X[idx]
            else:
                X_train_svm = X

            svm = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
            svm.fit(X_train_svm)
            # Negative scores (higher = more anomalous)
            svm_scores = -svm.score_samples(X)
            exec_time = time.time() - start

            result = BenchmarkResult(
                method_name="One-Class SVM",
                scores=svm_scores,
                labels=labels,
                execution_time=exec_time,
            )

            if labels is not None:
                result.auc_roc = roc_auc_score(labels, svm_scores)
                result.auc_pr = average_precision_score(labels, svm_scores)
                self._log(f"  AUC-ROC: {result.auc_roc:.3f}, AUC-PR: {result.auc_pr:.3f}")

            results["one_class_svm"] = result

        self.results_history.append(results)
        return results

    def evaluate_uncertainty(
        self,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare uncertainty estimation methods.

        Requires a model with predict_proba support.

        Parameters
        ----------
        X : array-like
            Test features
        y_true : array-like, optional
            Ground truth labels for calibration assessment
        methods : list of str, optional
            Methods: ['sentinel', 'entropy', 'margin', 'confidence']
        """
        if self.model is None or not hasattr(self.model, "predict_proba"):
            raise ValueError("Model with predict_proba required for uncertainty evaluation")

        methods = methods or ["sentinel", "entropy", "margin", "confidence"]
        X = np.asarray(X)
        results = {}

        self._log(f"Running uncertainty benchmark on {len(X)} samples...")

        # Get predictions
        probs = self.model.predict_proba(X)

        # SentinelML (trust as uncertainty inverse)
        if "sentinel" in methods:
            sentinel_uncertainty = []
            for x in X:
                result = self.sentinel.assess(x)
                # Uncertainty = 1 - trust
                sentinel_uncertainty.append(1.0 - result.trust_score)

            scores = np.array(sentinel_uncertainty)
            result = BenchmarkResult(
                method_name="SentinelML (Trust)",
                scores=scores,
                metadata={"interpretation": "1 - trust_score"},
            )

            if y_true is not None:
                preds = self.model.predict(X)
                errors = (preds != y_true).astype(int)
                if len(np.unique(errors)) > 1:
                    result.auc_roc = roc_auc_score(errors, scores)

            results["sentinel"] = result

        # Entropy
        if "entropy" in methods:
            # Max normalized entropy
            entropies = np.array([entropy(p) for p in probs])
            max_entropy = np.log(probs.shape[1])
            normalized_entropy = entropies / max_entropy

            result = BenchmarkResult(
                method_name="Prediction Entropy",
                scores=normalized_entropy,
                metadata={"max_entropy": max_entropy},
            )

            if y_true is not None:
                preds = self.model.predict(X)
                errors = (preds != y_true).astype(int)
                if len(np.unique(errors)) > 1:
                    result.auc_roc = roc_auc_score(errors, normalized_entropy)

            results["entropy"] = result

        # Margin (confidence gap)
        if "margin" in methods:
            sorted_probs = np.sort(probs, axis=1)
            margins = 1 - (sorted_probs[:, -1] - sorted_probs[:, -2])

            result = BenchmarkResult(
                method_name="Margin (Confidence Gap)",
                scores=margins,
                metadata={"interpretation": "1 - (p_max - p_second)"},
            )

            if y_true is not None:
                preds = self.model.predict(X)
                errors = (preds != y_true).astype(int)
                if len(np.unique(errors)) > 1:
                    result.auc_roc = roc_auc_score(errors, margins)

            results["margin"] = result

        # Confidence (1 - max prob)
        if "confidence" in methods:
            confidence = 1 - np.max(probs, axis=1)

            result = BenchmarkResult(
                method_name="1 - Max Confidence",
                scores=confidence,
                metadata={"interpretation": "1 - max(probabilities)"},
            )

            if y_true is not None:
                preds = self.model.predict(X)
                errors = (preds != y_true).astype(int)
                if len(np.unique(errors)) > 1:
                    result.auc_roc = roc_auc_score(errors, confidence)

            results["confidence"] = result

        self.results_history.append(results)
        return results

    def evaluate_drift_detection(
        self,
        X_ref: np.ndarray,
        X_test: np.ndarray,
        drift_labels: Optional[np.ndarray] = None,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare drift detection methods.

        Parameters
        ----------
        X_ref : array-like
            Reference data
        X_test : array-like
            Test data (may contain drift)
        drift_labels : array-like, optional
            Binary labels indicating drift points
        methods : list of str, optional
            Methods: ['sentinel', 'ks_test', 'adwin']
        """
        methods = methods or ["sentinel", "ks_test"]
        X_ref = np.asarray(X_ref)
        X_test = np.asarray(X_test)
        results = {}

        self._log(f"Running drift detection benchmark...")
        self._log(f"  Reference: {len(X_ref)} samples, Test: {len(X_test)} samples")

        # SentinelML
        if "sentinel" in methods:
            self._log("Evaluating SentinelML drift detection...")

            # Ensure sentinel is fitted on reference
            if hasattr(self.sentinel, "is_fitted") and not self.sentinel.is_fitted:
                self.sentinel.fit(X_ref)
            elif hasattr(self.sentinel.drift_detector, "fit"):
                self.sentinel.drift_detector.fit(X_ref)

            drift_scores = []
            for x in X_test:
                result = self.sentinel.assess(x)
                # Use drift p-value as score (lower = more drift)
                drift_scores.append(1.0 - getattr(result, "drift_pvalue", 0.5))

            scores = np.array(drift_scores)
            result = BenchmarkResult(
                method_name="SentinelML Drift",
                scores=scores,
                labels=drift_labels,
                metadata={"interpretation": "1 - p_value"},
            )

            if drift_labels is not None and len(np.unique(drift_labels)) > 1:
                result.auc_roc = roc_auc_score(drift_labels, scores)

            results["sentinel"] = result

        # KS Test (per-feature, aggregated)
        if "ks_test" in methods:
            self._log("Evaluating Kolmogorov-Smirnov test...")

            ks_scores = []
            for x in X_test:
                # Compare against reference using KS test
                p_values = []
                for feat_idx in range(X_ref.shape[1]):
                    _, p_val = ks_2samp(X_ref[:, feat_idx], [x[feat_idx]])
                    p_values.append(p_val)
                # Aggregate: minimum p-value (most significant drift)
                ks_scores.append(1.0 - np.min(p_values))

            scores = np.array(ks_scores)
            result = BenchmarkResult(
                method_name="Kolmogorov-Smirnov",
                scores=scores,
                labels=drift_labels,
                metadata={"aggregation": "min_pvalue"},
            )

            if drift_labels is not None and len(np.unique(drift_labels)) > 1:
                result.auc_roc = roc_auc_score(drift_labels, scores)

            results["ks_test"] = result

        self.results_history.append(results)
        return results

    def generate_report(
        self,
        results: Optional[Dict[str, BenchmarkResult]] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate comparison report as DataFrame.

        Parameters
        ----------
        results : dict, optional
            Results to report (defaults to last evaluation)
        output_path : str, optional
            Path to save CSV report

        Returns
        -------
        pd.DataFrame
            Comparison report
        """
        if results is None:
            if not self.results_history:
                raise ValueError("No results available. Run an evaluation first.")
            results = self.results_history[-1]

        rows = []
        for method_name, result in results.items():
            row = {
                "Method": result.method_name,
                "AUC-ROC": result.auc_roc,
                "AUC-PR": result.auc_pr,
                "Execution Time (s)": result.execution_time,
                "Mean Score": np.mean(result.scores),
                "Std Score": np.std(result.scores),
            }
            if result.metadata:
                row.update(result.metadata)
            rows.append(row)

        df = pd.DataFrame(rows)

        if output_path:
            df.to_csv(output_path, index=False)
            self._log(f"Report saved to {output_path}")

        return df

    def plot_comparison(
        self,
        results: Optional[Dict[str, BenchmarkResult]] = None,
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Any:
        """
        Plot comparison of methods.

        Parameters
        ----------
        results : dict, optional
            Results to plot
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        show : bool
            Whether to display plot

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting. Install: pip install matplotlib")

        if results is None:
            if not self.results_history:
                raise ValueError("No results available. Run an evaluation first.")
            results = self.results_history[-1]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 1. Score distributions
        ax = axes[0]
        for method_name, result in results.items():
            ax.hist(result.scores, bins=30, alpha=0.5, label=result.method_name, density=True)
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.set_title("Score Distributions")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Performance metrics (if available)
        ax = axes[1]
        methods = []
        auc_rocs = []
        auc_prs = []

        for method_name, result in results.items():
            if result.auc_roc is not None:
                methods.append(result.method_name)
                auc_rocs.append(result.auc_roc)
                auc_prs.append(result.auc_pr if result.auc_pr is not None else 0)

        if methods:
            x = np.arange(len(methods))
            width = 0.35
            ax.bar(x - width / 2, auc_rocs, width, label="AUC-ROC", alpha=0.8)
            ax.bar(x + width / 2, auc_prs, width, label="AUC-PR", alpha=0.8)
            ax.set_ylabel("Score")
            ax.set_title("Performance Metrics")
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.legend()
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(
                0.5,
                0.5,
                "No ground truth labels\nprovided for metrics",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Performance Metrics (N/A)")

        # 3. Execution time comparison
        ax = axes[2]
        methods = []
        times = []

        for method_name, result in results.items():
            if result.execution_time is not None:
                methods.append(result.method_name)
                times.append(result.execution_time)

        if methods:
            colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
            bars = ax.barh(methods, times, color=colors, alpha=0.8)
            ax.set_xlabel("Execution Time (s)")
            ax.set_title("Computational Efficiency")
            ax.grid(True, alpha=0.3, axis="x")

            # Add value labels
            for bar, time in zip(bars, times):
                width = bar.get_width()
                ax.text(
                    width + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{time:.3f}s",
                    ha="left",
                    va="center",
                    fontsize=9,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "Timing data not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self._log(f"Plot saved to {save_path}")

        if show:
            plt.show()
        else:
            return fig


# Legacy class for backward compatibility
class LegacyBenchmarkComparison(BenchmarkComparison):
    """
    Legacy benchmarking class for backward compatibility with v1.0 API.

    Maintains the simple interface from SentinelML v1.0 while using
    the new v2.0 benchmarking infrastructure internally.
    """

    def evaluate(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Simple evaluation matching v1.0 API.

        Parameters
        ----------
        X : array-like
            Test data
        y : array-like, optional
            Labels (not used, kept for compatibility)

        Returns
        -------
        dict
            Dictionary with keys: 'sentinel', 'entropy', 'isolation_forest', 'lof'
        """
        X = np.asarray(X)

        # Get model predictions if available
        if self.model is not None and hasattr(self.model, "predict_proba"):
            preds = self.model.predict_proba(X)
            entropy_scores = np.array([entropy(p) for p in preds])
        else:
            entropy_scores = np.zeros(len(X))

        # Sentinel scores
        trust_scores = []
        for x in X:
            result = self.sentinel.assess(x)
            trust_scores.append(result.trust_score)
        trust_scores = np.array(trust_scores)

        # Isolation Forest
        iso = IsolationForest(random_state=self.random_state).fit(X)
        iso_scores = -iso.score_samples(X)

        # LOF
        lof = LocalOutlierFactor(novelty=True)
        lof.fit(X)
        lof_scores = -lof.score_samples(X)

        return {
            "sentinel": trust_scores,
            "entropy": entropy_scores,
            "isolation_forest": iso_scores,
            "lof": lof_scores,
        }


# Convenience function for quick benchmarking
def quick_benchmark(
    sentinel,
    X: np.ndarray,
    model: Optional[Any] = None,
    y_true: Optional[np.ndarray] = None,
    contamination: float = 0.1,
) -> pd.DataFrame:
    """
    Quick benchmark function for common use case.

    Parameters
    ----------
    sentinel : Sentinel
        Configured Sentinel instance
    X : array-like
        Test data
    model : Any, optional
        Model with predict_proba
    y_true : array-like, optional
        Ground truth for metrics
    contamination : float
        Expected anomaly proportion

    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    bench = BenchmarkComparison(sentinel, model=model, verbose=False)

    # Run anomaly detection benchmark
    results = bench.evaluate_anomaly_detection(X, labels=y_true, contamination=contamination)

    # Add uncertainty evaluation if model available
    if model is not None and hasattr(model, "predict_proba"):
        unc_results = bench.evaluate_uncertainty(X, y_true=y_true)
        results.update(unc_results)

    return bench.generate_report(results)


__all__ = ["BenchmarkComparison", "BenchmarkResult", "LegacyBenchmarkComparison", "quick_benchmark"]
