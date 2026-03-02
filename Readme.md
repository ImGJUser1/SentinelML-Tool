# 📁  Project Structure

sentinelml/
├── __init__.py
├── __main__.py
├── cli.py
├── viz.py
├── core.py
├── drift.py
├── familiarity.py
├── incremental.py
├── plugins.py
├── state.py
├── trust.py
├── utils.py
├── adapters/
│   └── torch_adapter.py
├── benchmarks/
│   └── evaluator.py

pyproject.toml
README.md
```
## SentinelML: A Runtime Trust Layer for Machine Learning Systems

### Abstract

Modern machine learning systems assume that training and deployment data share identical statistical properties. In practice, this assumption fails due to distributional drift, novel inputs, and sensor instability. SentinelML introduces a lightweight runtime trust layer that operates alongside predictive models to estimate whether a model should be trusted on a per-input basis. Unlike monitoring platforms, SentinelML is an embeddable library requiring no infrastructure.

---

### 1. Introduction

Machine learning evaluation traditionally focuses on aggregate metrics such as accuracy or F1 score. These metrics fail to describe model reliability during deployment, where inputs may deviate significantly from the training distribution.

SentinelML addresses this gap by introducing:

• Familiarity estimation — is this input similar to training data?
• Statistical drift detection — has the environment changed?
• Geometric trust scoring — does this sample lie within learned structure?

---

### 2. Design Goals

SentinelML is designed to:

1. Be embeddable like a numerical library.
2. Require no services or orchestration.
3. Operate online with millisecond latency.
4. Provide interpretable trust signals rather than opaque alerts.

---

### 3. Method

Given training data (X), SentinelML constructs three concurrent estimators:

#### 3.1 Familiarity Model

A KD-tree estimates local density. Trust decreases exponentially with nearest-neighbor distance.

[
T_f(x) = e^{-d(x,X)/\sigma}
]

---

#### 3.2 Distribution Drift

A rolling window is compared against the reference dataset using the Kolmogorov–Smirnov test across features.

[
D = \min_i p_i
]

Drift is flagged when (D < \alpha).

---

#### 3.3 Geometric Trust

Mahalanobis distance evaluates whether the input lies within the covariance structure:

[
T_g(x) = e^{-\sqrt{(x-\mu)^T \Sigma^{-1} (x-\mu)}}
]

---

### 4. Unified Trust Score

The final trust score is a weighted fusion:

[
T(x) = 0.6T_f(x) + 0.4T_g(x)
]

This balances local familiarity with global structure.

---

### 5. Implementation

SentinelML is implemented in under 2,000 lines of Python and integrates with any predictive model without modification.

---

### 6. Use Cases

• Industrial sensor validation
• Medical decision support safeguards
• Autonomous system anomaly detection
• Financial regime shift monitoring

---

### 7. Conclusion

SentinelML reframes ML deployment as a runtime validation problem rather than a static evaluation problem, providing a missing reliability layer between models and real-world environments.

