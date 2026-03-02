import numpy as np
from .familiarity import FamiliarityModel
from .drift import DriftDetector
from .trust import TrustModel
from .utils import ensure_2d

class Sentinel:
    def __init__(self, drift_alpha=0.01, window_size=256):
        self.familiarity = FamiliarityModel()
        self.drift = DriftDetector(drift_alpha, window_size)
        self.trust = TrustModel()
        self.plugins = []
        self.reference = None

    def fit(self, X):
        X = ensure_2d(X)
        self.reference = X
        self._refit(X)

    def _refit(self, X):
        self.familiarity.fit(X)
        self.drift.fit(X)
        self.trust.fit(X)

        for p in self.plugins:
            p.fit(X)

    def register_plugin(self, plugin):
        plugin.fit(self.reference)
        self.plugins.append(plugin)

    def assess(self, x):
        x = np.asarray(x)

        fam_score = self.familiarity.score(x)
        trust_score = self.trust.score(x)

        self.drift.update(x)
        drift_flag, drift_p = self.drift.check()

        plugin_scores = [p.score(x) for p in self.plugins]
        if plugin_scores:
            trust_score = 0.5 * trust_score + 0.5 * np.mean(plugin_scores)

        final_trust = 0.6 * fam_score + 0.4 * trust_score

        return {
            "trust": float(final_trust),
            "familiarity": float(fam_score),
            "drift_detected": drift_flag,
            "drift_p_value": drift_p
        }