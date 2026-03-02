import numpy as np
from scipy.stats import ks_2samp
from .utils import ensure_2d

class DriftDetector:
    def __init__(self, alpha=0.01, window_size=256):
        self.alpha = alpha
        self.window_size = window_size
        self.reference = None
        self.window = []

    def fit(self, X):
        self.reference = ensure_2d(X)

    def update(self, x):
        self.window.append(x)
        if len(self.window) > self.window_size:
            self.window.pop(0)

    def check(self):
        if len(self.window) < 30:
            return False, 0.0

        window = np.array(self.window)
        scores = []

        for i in range(self.reference.shape[1]):
            stat, p = ks_2samp(self.reference[:, i], window[:, i])
            scores.append(p)

        min_p = min(scores)
        return min_p < self.alpha, float(min_p)