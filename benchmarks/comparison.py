import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import entropy


class BenchmarkComparison:

    def __init__(self, sentinel, model):
        self.sentinel = sentinel
        self.model = model

    def evaluate(self, X, y):

        trust_scores = []
        entropy_scores = []

        preds = self.model.predict_proba(X)

        for x, p in zip(X, preds):

            trust_scores.append(self.sentinel.assess(x)["trust"])
            entropy_scores.append(entropy(p))

        trust_scores = np.array(trust_scores)
        entropy_scores = np.array(entropy_scores)

        iso = IsolationForest().fit(X)
        iso_scores = -iso.score_samples(X)

        lof = LocalOutlierFactor(novelty=True)
        lof.fit(X)
        lof_scores = -lof.score_samples(X)

        return {
            "sentinel": trust_scores,
            "entropy": entropy_scores,
            "isolation_forest": iso_scores,
            "lof": lof_scores
        }