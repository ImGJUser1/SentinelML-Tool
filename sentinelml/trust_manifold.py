import numpy as np
from sklearn.neighbors import NearestNeighbors


class AdaptiveTrustManifold:
    def __init__(self, k=5):
        self.k = k
        self.model = NearestNeighbors(n_neighbors=k)
        self.reference = None

    def fit(self, predictions):
        self.reference = predictions
        self.model.fit(predictions)

    def trust_score(self, new_preds):
        distances, _ = self.model.kneighbors(new_preds)
        score = np.exp(-np.mean(distances))
        return score
