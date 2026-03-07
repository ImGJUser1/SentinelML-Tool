from sentinelml.core import Sentinel
import numpy as np

def test_basic_trust():
    X = np.random.rand(100, 4)

    s = Sentinel()
    s.fit(X)

    result = s.assess(X[0])

    assert "trust" in result
    assert result["trust"] >= 0