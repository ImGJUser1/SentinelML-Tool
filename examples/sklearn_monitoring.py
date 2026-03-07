import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sentinelml import SentinelMonitor

# train model
X = np.random.randn(1000, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = RandomForestClassifier()
model.fit(X, y)

monitor = SentinelMonitor()

# simulate production predictions
X_test = np.random.randn(100, 5)
preds = model.predict_proba(X_test)

report = monitor.analyze(preds)

print(report)
