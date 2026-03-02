import numpy as np

class TrustModel:
    def __init__(self):
        self.mean = None
        self.cov_inv = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        cov = np.cov(X.T) + np.eye(X.shape[1]) * 1e-6
        self.cov_inv = np.linalg.inv(cov)

    def mahalanobis(self, x):
        diff = x - self.mean
        return np.sqrt(diff @ self.cov_inv @ diff.T)

    def score(self, x):
        d = self.mahalanobis(x)
        return float(np.exp(-d))