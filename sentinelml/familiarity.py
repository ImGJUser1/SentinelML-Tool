import numpy as np
from sklearn.neighbors import KDTree
from .utils import ensure_2d

class FamiliarityModel:
    def __init__(self):
        self.tree = None
        self.reference = None
        self.scale = None

    def fit(self, X):
        X = ensure_2d(X)
        self.reference = X
        self.tree = KDTree(X)
        dists, _ = self.tree.query(X, k=2)
        self.scale = np.median(dists[:, 1]) + 1e-8

    def score(self, x):
        x = ensure_2d(x)
        dist, _ = self.tree.query(x, k=1)
        score = np.exp(-(dist / self.scale))
        return float(score[0][0])