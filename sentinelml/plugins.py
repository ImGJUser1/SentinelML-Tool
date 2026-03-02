class TrustPlugin:
    def fit(self, X):
        raise NotImplementedError

    def score(self, x):
        raise NotImplementedError