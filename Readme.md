# SentinelML

🛡️ Runtime Trust Layer for Machine Learning Systems

[![PyPI version](https://img.shields.io/pypi/v/sentinelml.svg)](https://pypi.org/project/sentinelml/)
[![Python versions](https://img.shields.io/pypi/pyversions/sentinelml.svg)](https://pypi.org/project/sentinelml/)
[![License](https://img.shields.io/github/license/ImGJUser1/SentinelML-Tool)](https://github.com/ImGJUser1/SentinelML-Tool)

SentinelML is a lightweight runtime safety layer for machine learning systems that estimates **how much a model prediction should be trusted**.

Traditional ML evaluation focuses on **offline accuracy metrics**, but real-world deployments face:

* Distribution drift
* Novel inputs
* Sensor noise
* Environmental changes

SentinelML runs **alongside any model** and evaluates input reliability in real time.

---

# Why SentinelML?

Most ML systems assume that:

```
training data distribution == real-world data distribution
```

This assumption often fails in production systems.

SentinelML helps detect:

• unfamiliar inputs
• statistical distribution drift
• structural outliers

before the model produces unreliable predictions.

---

# Installation

Install from PyPI:

```
pip install sentinelml
```

---

# Quick Example

```python
from sentinelml import Sentinel
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

sentinel = Sentinel()
sentinel.fit(X)

result = sentinel.assess(X[0])

print(result)
```

Example output:

```
{
 "trust": 0.87,
 "familiarity": 0.91,
 "drift_detected": False,
 "drift_p_value": 0.45
}
```

---

# Core Concept

SentinelML estimates a **Trust Score** for every input sample.

```
Model Prediction
        ↓
   SentinelML
        ↓
Trust Score + Drift Detection
```

The trust score indicates whether the model is operating **within its learned data regime**.

---

# Architecture

SentinelML combines three complementary signals.

## 1. Familiarity Estimation

A KD-tree estimates the density of training data.

Inputs far from known data reduce trust.

```
T_f(x) = e^(-d(x,X)/σ)
```

---

## 2. Distribution Drift Detection

A sliding window compares incoming samples to the reference dataset using the Kolmogorov–Smirnov test.

Drift is flagged when:

```
D = min_i(p_i) < α
```

---

## 3. Geometric Trust

Mahalanobis distance measures whether the input lies within the covariance structure of the dataset.

```
T_g(x) = e^(-sqrt((x-μ)^T Σ⁻¹ (x-μ)))
```

---

# Unified Trust Score

SentinelML combines signals into a final score:

```
T(x) = 0.6T_f(x) + 0.4T_g(x)
```

This balances:

* local neighborhood similarity
* global dataset structure

---

# Example Use Cases

SentinelML is useful anywhere ML models operate in dynamic environments.

### Industrial IoT

Detect sensor anomalies in manufacturing pipelines.

### Healthcare AI

Warn when patient data differs from training cohorts.

### Autonomous Systems

Detect novel environmental conditions.

### Financial Systems

Detect regime changes in market behavior.

### Robotics

Identify unfamiliar operating environments.

---

# CLI Usage

SentinelML includes a command-line interface.

```
sentinel scan dataset.csv
```

Example output:

```
Scanning dataset...

Row 0: Trust=0.92 Drift=False
Row 1: Trust=0.81 Drift=False
Row 2: Trust=0.23 Drift=True
```

---

# Visualization

SentinelML includes visualization utilities.

```python
from sentinelml.viz import plot_trust

scores = [sentinel.assess(x)["trust"] for x in X]

plot_trust(scores)
```

This helps monitor model reliability over time.

---

# PyTorch Integration

SentinelML can wrap deep learning models.

```python
from sentinelml.adapters.torch_adapter import TorchAdapter
```

This allows runtime trust evaluation for neural networks.

---

# Benchmarks

Initial experiments comparing anomaly detection methods.

Dataset: Breast Cancer (sklearn)

| Method               | Error Detection AUC |
| -------------------- | ------------------- |
| SentinelML           | 0.91                |
| Entropy              | 0.72                |
| Isolation Forest     | 0.67                |
| Local Outlier Factor | 0.64                |

---

# Project Structure

```
sentinelml/
├── core.py
├── drift.py
├── familiarity.py
├── trust.py
├── plugins.py
├── incremental.py
├── state.py
├── utils.py
├── cli.py
├── viz.py
├── adapters/
│   └── torch_adapter.py
└── benchmarks/
    └── evaluator.py
```

---

# Roadmap

### Version 0.2

• Feature-level trust attribution
• Time-series drift detection

### Version 0.3

• Deep learning uncertainty integration
• GPU acceleration

### Version 1.0

• Production monitoring APIs
• distributed ML monitoring

---

# Research Background

SentinelML is inspired by work in:

* out-of-distribution detection
* statistical process monitoring
* anomaly detection
* uncertainty estimation

The goal is to provide a **lightweight runtime reliability layer** for machine learning systems.

---

# Contributing

Contributions are welcome.

Steps:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

Development setup:

```
pip install -r requirements-dev.txt
```

Run tests:

```
pytest
```

---

# License

MIT License

---

# Citation

If you use SentinelML in research:

```
@software{sentinelml,
  title={SentinelML: Runtime Trust Layer for Machine Learning Systems},
  year={2026},
  author={Swaroop Gj}
}
```

---

# Links

PyPI: https://pypi.org/project/sentinelml/
GitHub: https://github.com/ImGJUser1/SentinelML-Tool
