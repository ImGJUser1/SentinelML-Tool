from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/traditional/drift/adversarial.py
"""
Adversarial drift detection using learned discriminators.
"""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from sentinelml.traditional.drift.base import BaseDriftDetector


class AdversarialDriftDetector(BaseDriftDetector):
    """
    Detects drift by training a classifier to distinguish
    reference from new data. If classifier can distinguish them
    better than random, drift has occurred.

    Parameters
    ----------
    classifier : object, optional
        Binary classifier (defaults to RandomForest).
    n_splits : int, default=5
        Number of CV splits for evaluation.
    test_size : float, default=0.2
        Fraction for test set.

    Examples
    --------
    >>> detector = AdversarialDriftDetector()
    >>> detector.fit(X_reference)
    >>> is_drift, p_values = detector.detect(X_new)
    """

    def __init__(
        self,
        name: str = "AdversarialDriftDetector",
        window_size: int = 1000,
        threshold: float = 0.05,
        classifier: Optional[Any] = None,
        n_splits: int = 5,
        test_size: float = 0.2,
        verbose: bool = False,
    ):
        super().__init__(name=name, window_size=window_size, threshold=threshold, verbose=verbose)
        self.classifier = classifier
        self.n_splits = n_splits
        self.test_size = test_size
        self._clf = None

    def fit(self, X: npt.ArrayLike, y: Optional[npt.NDArray] = None) -> "AdversarialDriftDetector":
        """Store reference data."""
        X = np.asarray(X)
        super().fit(X, y)

        # Initialize classifier
        if self.classifier is None:
            from sklearn.ensemble import RandomForestClassifier

            self._clf = RandomForestClassifier(n_estimators=100, max_depth=10)
        else:
            self._clf = self.classifier

        return self

    def detect(self, X: npt.ArrayLike) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """
        Train classifier to distinguish reference from new data.

        Returns
        -------
        is_drift : ndarray
            True if drift detected.
        p_values : ndarray
            1 - accuracy (lower = more drift).
        """
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Create labeled dataset
        n_ref = len(self.reference_data_)
        n_new = len(X)

        X_combined = np.vstack([self.reference_data_, X])
        y_combined = np.array([0] * n_ref + [1] * n_new)  # 0=ref, 1=new

        # Stratified CV to evaluate discriminability
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=min(self.n_splits, min(n_ref, n_new)), shuffle=True)
        scores = []

        for train_idx, test_idx in cv.split(X_combined, y_combined):
            X_train, X_test = X_combined[train_idx], X_combined[test_idx]
            y_train, y_test = y_combined[train_idx], y_combined[test_idx]

            clf = self._get_fresh_classifier()
            clf.fit(X_train, y_train)

            # Predict probabilities
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, probs)
            else:
                preds = clf.predict(X_test)
                auc = np.mean(preds == y_test)

            scores.append(auc)

        mean_auc = np.mean(scores)
        # Drift detected if AUC significantly > 0.5
        is_drift = mean_auc > (0.5 + self.threshold)
        p_value = 1.0 - mean_auc  # Lower = more drift-like

        n_samples = len(X)
        return (np.array([is_drift] * n_samples), np.array([p_value] * n_samples))

    def _get_fresh_classifier(self):
        """Get new classifier instance."""
        if self.classifier is None:
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        return self.classifier
