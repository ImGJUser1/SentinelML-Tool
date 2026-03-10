import numpy as np

class SentinelEvaluator:
    def __init__(self, sentinel, model):
        self.sentinel = sentinel
        self.model = model

    def run(self, X, y):
        trust_scores = []
        correct = []

        for xi, yi in zip(X, y):
            pred = self.model.predict([xi])[0]
            trust = self.sentinel.assess(xi)["trust"]

            trust_scores.append(trust)
            correct.append(int(pred == yi))

        trust_scores = np.array(trust_scores)
        correct = np.array(correct)

        high = correct[trust_scores > 0.7].mean()
        low = correct[trust_scores < 0.3].mean()

        return {
            "high_trust_accuracy": float(high),
            "low_trust_accuracy": float(low),
            "trust_separation": float(high - low)
        }
