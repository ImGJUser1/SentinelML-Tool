import numpy as np

class IncrementalUpdater:
    def __init__(self, sentinel, rate=0.01):
        self.sentinel = sentinel
        self.rate = rate

    def update(self, x):
        ref = self.sentinel.reference
        idx = np.random.randint(len(ref))
        ref[idx] = (1 - self.rate) * ref[idx] + self.rate * x

        self.sentinel._refit(ref)