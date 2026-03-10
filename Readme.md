# SentinelML

[![PyPI version](https://badge.fury.io/py/sentinelml.svg)](https://pypi.org/project/sentinelml/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Unified Reliability Engine for AI/ML Systems**

SentinelML is a comprehensive framework for monitoring, evaluating, and ensuring the reliability of machine learning systems across traditional ML, deep learning, generative AI, RAG pipelines, and agentic systems.

---

## 🚀 Features

### Multi-Domain Support
- **Traditional ML**: Drift detection, anomaly detection, out-of-distribution detection
- **Deep Learning**: Uncertainty quantification, adversarial detection, feature drift monitoring
- **Generative AI**: Input/output guardrails, hallucination detection, bias detection
- **RAG Systems**: Retrieval relevance, faithfulness checking, end-to-end evaluation (RAGAS, ARES)
- **Agent Systems**: Trajectory validation, tool monitoring, reasoning consistency

### Core Capabilities
| Capability | Description |
|------------|-------------|
| 🔍 **Drift Detection** | KS-test, PSI, MMD, Adversarial drift detectors |
| 🛡️ **Trust Scoring** | Mahalanobis distance, Isolation Forest, VAE-based anomaly detection |
| 🎯 **Uncertainty Quantification** | MC Dropout, Deep Ensembles, Evidential Networks, Temperature Scaling |
| 🔒 **Guardrails** | Prompt injection detection, PII filtering, toxicity detection, schema validation |
| 📊 **Visualization** | Trust dashboards, drift plots, interactive Plotly dashboards |
| 🖥️ **Serving** | FastAPI and gRPC servers for production monitoring |

---

## 📦 Installation

```bash
# Basic installation (Traditional ML only)
pip install sentinelml

# With PyTorch support
pip install sentinelml[torch]

# With TensorFlow support
pip install sentinelml[tensorflow]

# For Generative AI / LLM applications
pip install sentinelml[genai]

# For RAG applications
pip install sentinelml[rag]

# For production serving
pip install sentinelml[serving]

# Complete installation
pip install sentinelml[all]

# Development installation
pip install sentinelml[dev]
```

---

## 🏃 Quick Start

### Traditional ML Monitoring

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sentinelml import Sentinel, KSDriftDetector, MahalanobisTrust

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test = X[:100], X[100:]

# Train your model
model = RandomForestClassifier().fit(X_train, y[:100])

# Initialize Sentinel with drift and trust monitoring
sentinel = Sentinel(
    drift_detector=KSDriftDetector(threshold=0.05),
    trust_model=MahalanobisTrust(),
    verbose=True
)

# Fit on reference (training) data
sentinel.fit(X_train)

# Assess new samples
results = []
for x in X_test:
    result = sentinel.assess(x)
    results.append(result)
    print(f"Trust: {result.trust_score:.3f}, Drift: {result.has_drift}")

# Visualize
from sentinelml.viz import plot_trust
trust_scores = [r.trust_score for r in results]
plot_trust(trust_scores, title="Trust Scores on Test Data")
```

### Detecting Drift

```python
import numpy as np

# Simulate drifted data
drift_data = X_test + np.random.normal(0, 2, X_test.shape)

# Assess drifted samples
for x in drift_data[:5]:
    result = sentinel.assess(x)
    print(f"Trust: {result.trust_score:.3f}, "
          f"Drift p-value: {result.drift_pvalue:.4f}, "
          f"Is Trustworthy: {result.is_trustworthy}")
```

### GenAI Guardrails

```python
from sentinelml import PromptInjectionDetector, HallucinationDetector

# Input validation
injection_detector = PromptInjectionDetector(threshold=0.7)
result = injection_detector.detect("Ignore previous instructions and...")
print(f"Injection detected: {result.is_violation}, Score: {result.score}")

# Output validation (RAG context)
hallucination_detector = HallucinationDetector(method="self_consistency")
context = ["Paris is the capital of France.", "France is in Europe."]
generated = "Paris is the capital of Germany."
result = hallucination_detector.verify(context, generated)
print(f"Hallucination detected: {result.is_hallucination}")
```

### RAG Evaluation

```python
from sentinelml import RAGASEvaluator, FaithfulnessChecker

# End-to-end RAG evaluation
evaluator = RAGASEvaluator(metrics=["faithfulness", "answer_relevancy", "context_recall"])
results = evaluator.evaluate(
    questions=["What is the capital of France?"],
    answers=["Paris is the capital of France."],
    contexts=[["Paris is the capital of France."]],
    ground_truths=["Paris"]
)

# Component-level checking
faithfulness = FaithfulnessChecker()
score = faithfulness.check(answer="Paris is the capital.", context="Paris is France's capital city.")
```

### Agent Monitoring

```python
from sentinelml import StepValidator, LoopDetector, BudgetManager

# Monitor agent execution
validator = StepValidator()
loop_detector = LoopDetector(window_size=5)
budget = BudgetManager(max_steps=50, max_tokens=10000)

# Validate each step
for step_num, (thought, action, observation) in enumerate(agent_steps):
    validation = validator.validate_step(thought, action, observation)
    if loop_detector.detect_loop(agent_steps[:step_num+1]):
        print("Loop detected! Breaking...")
        break
    if not budget.consume_step(tokens_used=len(thought.split())):
        print("Budget exceeded!")
        break
```

---

## 🖥️ Command Line Interface

```bash
# Scan dataset for drift and anomalies
sentinelml scan data.csv --drift-detector mmd --trust-model mahalanobis --output report.json

# Evaluate model reliability
sentinelml evaluate model.pkl test.csv --labels target --output evaluation.json

# Start monitoring server
sentinelml serve --port 8000 --config sentinel.yaml

# Generate configuration template
sentinelml config --type genai --output sentinel.yaml
```

---

## 📁 Project Structure (v2.0)

```
sentinelml/
├── core/                    # Core engine and orchestration
│   ├── sentinel.py         # Main Sentinel orchestrator
│   ├── pipeline.py         # Processing pipelines
│   ├── ensemble.py         # Adaptive trust ensembles
│   └── report.py           # Reporting infrastructure
├── traditional/            # Traditional ML monitoring
│   ├── drift/             # Drift detection methods
│   ├── trust/             # Anomaly/trust scoring
│   └── familiarity/       # OOD detection
├── deep_learning/         # Deep learning specific
│   ├── uncertainty/       # UQ methods (MC Dropout, Ensembles, etc.)
│   ├── feature_drift/     # Activation/embedding monitoring
│   └── adversarial/       # Adversarial attack detection
├── genai/                 # Generative AI guardrails
│   ├── guardrails/        # Input/output validation
│   ├── alignment/         # Bias and toxicity detection
│   └── uncertainty/       # LLM uncertainty estimation
├── rag/                   # RAG pipeline evaluation
│   ├── retrieval/         # Retrieval metrics
│   ├── generation/        # Generation quality
│   ├── advanced/          # Claim verification, contradiction detection
│   └── end_to_end/        # RAGAS, ARES evaluators
├── agents/                # Agent system monitoring
│   ├── trajectory/        # Step validation, loop detection
│   ├── reasoning/         # Logic checking, consistency
│   └── state/             # Budget and checkpoint management
├── adapters/              # Framework integrations
│   ├── sklearn_adapter.py
│   ├── torch_adapter.py
│   ├── tensorflow_adapter.py
│   ├── openai_adapter.py
│   ├── langchain_adapter.py
│   └── ...
├── infrastructure/        # Production infrastructure
│   ├── serving/           # FastAPI/gRPC servers
│   ├── storage/           # Vector store integration
│   └── streaming/         # Kafka consumers
└── viz.py                # Visualization utilities
```

---

## 🔧 Configuration

Create a configuration file for different deployment scenarios:

```yaml
# sentinel.yaml - Traditional ML
sentinel:
  drift_detector:
    type: mmd
    threshold: 0.05
  trust_model:
    type: mahalanobis
    calibration: isotonic

monitoring:
  batch_size: 1000
  check_interval: 3600
```

```yaml
# sentinel.yaml - GenAI
sentinel:
  guardrails:
    input:
      - type: prompt_injection
        threshold: 0.7
      - type: pii_detection
        entities: [email, phone, ssn]
    output:
      - type: hallucination_detection
        method: self_consistency

llm:
  model: gpt-4
  temperature: 0.7
```

---

## 📊 Benchmarking & Comparison

SentinelML includes comprehensive benchmarking tools to compare against baseline methods:

```python
from sentinelml.benchmarks import BenchmarkComparison
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Compare Sentinel against baselines
benchmark = BenchmarkComparison(sentinel=sentinel, model=model)
results = benchmark.evaluate(X_test, y_test)

# Returns comparison of:
# - sentinel: Trust scores from SentinelML
# - entropy: Prediction entropy (uncertainty)
# - isolation_forest: Isolation Forest anomaly scores
# - lof: Local Outlier Factor scores
```

---

## 🛣️ Roadmap

### Version 2.1 (Current)
- ✅ Modular architecture rewrite
- ✅ GenAI guardrails (input/output)
- ✅ RAG evaluation framework
- ✅ Agent monitoring tools
- ✅ FastAPI/gRPC serving

### Version 2.2 (Upcoming)
- Streaming drift detection (Kafka integration)
- Distributed monitoring (Ray/Spark)
- Advanced attribution methods
- Automated threshold tuning

### Version 3.0 (Future)
- Multi-modal support (vision, audio)
- Real-time adversarial defense
- LLM-powered root cause analysis
- Enterprise dashboard

---

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md).

```bash
# Development setup
git clone https://github.com/sentinelml/sentinelml.git
cd sentinelml
pip install -e ".[dev]"

# Run tests
pytest tests/ --cov=sentinelml

# Code quality
black sentinelml/ tests/
isort sentinelml/ tests/
flake8 sentinelml/ tests/
```

---

## 📚 Research Background

SentinelML integrates research from:

- **Out-of-Distribution Detection**: Hendrycks & Gimpel, Liu et al.
- **Drift Detection**: Rabanser et al. (MMD), dos Reis et al. (PSI)
- **Uncertainty Quantification**: Gal & Ghahramani (MC Dropout), Lakshminarayanan et al. (Deep Ensembles)
- **LLM Safety**: Perez & Ribeiro (red teaming), Minding the Gap (hallucination detection)
- **RAG Evaluation**: Es et al. (RAGAS), Saad-Falcon et al. (ARES)

---

## 📄 Citation

If you use SentinelML in your research:

```bibtex
@software{sentinelml2024,
  title={SentinelML: Unified Reliability Engine for AI/ML Systems},
  author={SentinelML Team},
  year={2024},
  version={2.0.0},
  url={https://github.com/sentinelml/sentinelml}
}
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🔗 Links

- **PyPI**: https://pypi.org/project/sentinelml/
- **Documentation**: https://sentinelml.readthedocs.io/
- **GitHub**: https://github.com/sentinelml/sentinelml
- **Issues**: https://github.com/sentinelml/sentinelml/issues

---

## 💡 Support

For questions and support:
- 📧 Email: team@sentinelml.ai
- 💬 Discussions: GitHub Discussions
- 🐛 Issues: GitHub Issues

---

**SentinelML**: *Trustworthy AI through continuous monitoring*
