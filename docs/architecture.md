# SentinelML Architecture Documentation

> **Version 2.0.0** | Modular Reliability Engine for AI/ML Systems

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Domain-Specific Modules](#domain-specific-modules)
5. [Data Flow](#data-flow)
6. [Extension Points](#extension-points)
7. [Deployment Patterns](#deployment-patterns)
8. [Performance Considerations](#performance-considerations)

---

## Overview

SentinelML v2.0 is built on a **layered, modular architecture** that separates concerns across six distinct domains:

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│  (CLI, Notebooks, FastAPI/gRPC Servers, Streaming)         │
├─────────────────────────────────────────────────────────────┤
│                    ORCHESTRATION LAYER                       │
│  (Sentinel, Pipeline, Ensemble, Report)                      │
├─────────────────────────────────────────────────────────────┤
│                    DOMAIN LAYER                              │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │Traditional│  Deep   │  GenAI  │   RAG   │  Agents │   │
│  │    ML    │Learning │         │         │         │   │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    ADAPTER LAYER                             │
│  (sklearn, torch, tensorflow, openai, langchain, etc.)     │
├─────────────────────────────────────────────────────────────┤
│                    INFRASTRUCTURE LAYER                      │
│  (Serving, Storage, Streaming)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## System Architecture

### Design Principles

1. **Modularity**: Each domain is self-contained with clear interfaces
2. **Composability**: Components can be mixed and matched via the Ensemble system
3. **Extensibility**: New detectors/adapters can be added without core changes
4. **Production-Ready**: Built-in serving, monitoring, and scaling capabilities

### High-Level Component Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                        Sentinel (Orchestrator)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Drift     │  │    Trust    │  │      Familiarity        │ │
│  │  Detector   │  │    Model    │  │       (OOD)             │ │
│  │  (Optional) │  │  (Optional) │  │      (Optional)         │ │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘ │
│         │                │                      │               │
│         └────────────────┼──────────────────────┘               │
│                          ▼                                      │
│                   ┌─────────────┐                               │
│                   │   Assess    │  → TrustReport                │
│                   │   Sample    │     (trust_score,             │
│                   │             │      has_drift,               │
│                   └─────────────┘      is_trustworthy)          │
└────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Sentinel (`sentinelml/core/sentinel.py`)

The main orchestration class that coordinates multiple monitoring strategies.

```python
class Sentinel:
    """
    Unified interface for ML reliability monitoring.

    Parameters
    ----------
    drift_detector : BaseDriftDetector, optional
        Drift detection strategy (e.g., MMDDriftDetector)
    trust_model : BaseTrustModel, optional
        Trust scoring strategy (e.g., MahalanobisTrust)
    familiarity_model : BaseFamiliarityModel, optional
        OOD detection strategy (e.g., HNSWFamiliarity)
    ensemble_strategy : str, default 'weighted'
        How to combine multiple signals
    verbose : bool, default False
        Enable logging
    """
```

**Key Methods:**
- `fit(X_ref)` - Calibrate on reference data
- `assess(x)` - Evaluate single sample
- `assess_batch(X)` - Evaluate batch efficiently
- `get_report()` - Generate comprehensive report

### 2. Pipeline (`sentinelml/core/pipeline.py`)

Sequential processing with conditional routing.

```python
pipeline = SentinelPipeline([
    ("drift_check", MMDDriftDetector()),
    ("trust_score", MahalanobisTrust(),
     condition=lambda x: not x.has_drift),  # Skip if drift detected
    ("guardrails", ToxicityFilter(),
     condition=lambda x: x.trust_score > 0.5)
])
```

### 3. Adaptive Trust Ensemble (`sentinelml/core/ensemble.py`)

Combines multiple trust signals with learned weights.

```
┌─────────────────────────────────────────┐
│         AdaptiveTrustEnsemble           │
│                                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │  Drift  │ │  Trust  │ │  OOD    │   │
│  │  Score  │ │  Score  │ │  Score  │   │
│  └────┬────┘ └────┬────┘ └────┬────┘   │
│       │           │           │         │
│       └───────────┼───────────┘         │
│                   ▼                     │
│            ┌────────────┐              │
│            │   Learned  │              │
│            │  Weights   │              │
│            │  [w1,w2,w3]│              │
│            └─────┬──────┘              │
│                  ▼                      │
│         Final Trust Score               │
└─────────────────────────────────────────┘
```

### 4. Reporting System (`sentinelml/core/report.py`)

Structured outputs for different monitoring scenarios.

| Report Type | Contents | Use Case |
|-------------|----------|----------|
| `TrustReport` | trust_score, confidence, components | Real-time monitoring |
| `DriftReport` | p_value, drift_detected, magnitude | Data quality alerts |
| `GuardrailReport` | violations, severity, action | Safety filtering |
| `RAGReport` | faithfulness, relevance, latency | RAG evaluation |

---

## Domain-Specific Modules

### 1. Traditional ML (`sentinelml/traditional/`)

**Drift Detection (`drift/`)**
```
BaseDriftDetector (ABC)
├── KSDriftDetector      - Kolmogorov-Smirnov test (univariate)
├── PSIDetector          - Population Stability Index
├── MMDDriftDetector     - Maximum Mean Discrepancy (multivariate)
└── AdversarialDriftDetector - Drift via adversarial validation
```

**Trust Models (`trust/`)**
```
BaseTrustModel (ABC)
├── MahalanobisTrust     - Statistical distance in feature space
├── IsolationForestTrust - Tree-based anomaly scores
├── VAETrust            - Reconstruction error from VAE
└── ConformalPredictor  - Conformal prediction sets
```

**Familiarity/OOD (`familiarity/`)**
```
BaseFamiliarityModel (ABC)
├── KDTreeFamiliarity    - K-nearest neighbors distance
├── HNSWFamiliarity      - Hierarchical Navigable SW graph (fast approximate)
└── KernelDensityFamiliarity - KDE-based density estimation
```

### 2. Deep Learning (`sentinelml/deep_learning/`)

**Uncertainty Quantification (`uncertainty/`)**
```
BaseUncertainty (ABC)
├── MCDropoutUncertainty    - Monte Carlo Dropout sampling
├── DeepEnsembleUncertainty - Multiple model predictions
├── TemperatureScaling      - Post-hoc calibration
└── EvidentialNetwork       - Evidential deep learning
```

**Feature Drift (`feature_drift/`)**
- `ActivationMonitor` - Track layer-wise activations
- `EmbeddingDrift` - Monitor embedding space shifts

**Adversarial Detection (`adversarial/`)**
- `FGSMDetector` - Fast Gradient Sign Method detection
- `BoundaryAttack` - Boundary-based adversarial detection

### 3. Generative AI (`sentinelml/genai/`)

**Input Guardrails (`guardrails/input/`)**
```
InputGuardrail (ABC)
├── PromptInjectionDetector - Detect jailbreak attempts
├── PIIDetector            - Identify personal information
├── ToxicityFilter         - Content moderation
└── IntentClassifier       - Validate user intent
```

**Output Guardrails (`guardrails/output/`)**
```
OutputGuardrail (ABC)
├── HallucinationDetector  - Factual consistency checking
├── ConsistencyCheck         - Self-consistency validation
├── SchemaValidator          - Structured output validation
└── CitationVerifier         - Source verification
```

**Uncertainty (`uncertainty/`)**
- `SemanticEntropy` - Entropy over semantic meaning
- `LexicalSimilarity` - N-gram overlap-based uncertainty
- `TokenLogProb` - Token-level probability aggregation

### 4. RAG Systems (`sentinelml/rag/`)

**Retrieval Metrics (`retrieval/`)**
- `RelevanceScorer` - Context-question relevance
- `CoverageAnalyzer` - Information coverage
- `DiversityMetrics` - Result diversity (MMR, etc.)

**Generation Metrics (`generation/`)**
- `FaithfulnessChecker` - Answer-context consistency
- `AnswerRelevance` - Answer-question alignment
- `CitationAccuracy` - Citation precision/recall

**End-to-End (`end_to_end/`)**
- `RAGASEvaluator` - RAGAS metrics implementation
- `ARESEvaluator` - ARES pairwise evaluation
- `LatencyTracker` - Performance monitoring

**Advanced (`advanced/`)**
- `ClaimVerification` - Fact-checking against sources
- `ContradictionDetect` - Cross-document consistency

### 5. Agent Systems (`sentinelml/agents/`)

**Trajectory Monitoring (`trajectory/`)**
- `StepValidator` - Per-step validation
- `ToolMonitor` - Tool call monitoring
- `LoopDetector` - Infinite loop prevention

**Reasoning Validation (`reasoning/`)**
- `LogicChecker` - Logical consistency
- `StepConsistency` - Cross-step coherence

**State Management (`state/`)**
- `BudgetManager` - Token/cost/step budgets
- `CheckpointManager` - Recovery points

---

## Data Flow

### Single Sample Assessment Flow

```
Input Sample x
│
├─► [Optional] Input Guardrails
│   └─► Check: Injection? PII? Toxicity?
│       └─► Violation? → Block / Log / Warn
│
├─► [Optional] Drift Detection
│   └─► Compare: x vs Reference Distribution
│       └─► p-value < threshold? → Drift Flag
│
├─► [Optional] Trust Scoring
│   └─► Compute: Anomaly Score / Distance
│       └─► Normalize to [0, 1]
│
├─► [Optional] Familiarity/OOD
│   └─► Compute: Density / Nearest Neighbor Distance
│       └─► OOD? → Low Familiarity Flag
│
├─► Ensemble (if multiple signals)
│   └─► Weighted combination
│
└─► Output: TrustReport
    ├─ trust_score: float [0-1]
    ├─ has_drift: bool
    ├─ is_trustworthy: bool
    ├─ component_scores: dict
    └─ metadata: dict
```

### Batch Processing Flow

```python
# Efficient batch assessment
sentinel = Sentinel(
    drift_detector=MMDDriftDetector(),
    trust_model=MahalanobisTrust()
)
sentinel.fit(X_ref)

# Vectorized where possible
results = sentinel.assess_batch(X_test, batch_size=1000)
```

**Optimization Strategies:**
1. **Vectorization**: NumPy operations on batches
2. **Caching**: Fit parameters stored, not recomputed
3. **Approximation**: HNSW for fast approximate NN
4. **Lazy Evaluation**: Compute only requested metrics

---

## Extension Points

### Adding a Custom Drift Detector

```python
from sentinelml.traditional.drift.base import BaseDriftDetector

class MyDriftDetector(BaseDriftDetector):
    def __init__(self, threshold=0.05, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.reference_stats = None

    def fit(self, X_ref):
        # Compute reference statistics
        self.reference_stats = {
            'mean': np.mean(X_ref, axis=0),
            'std': np.std(X_ref, axis=0)
        }
        self.is_fitted = True
        return self

    def detect(self, x):
        # Return DriftResult
        score = self._compute_score(x)
        p_value = self._compute_pvalue(score)
        return DriftResult(
            is_drift=p_value < self.threshold,
            p_value=p_value,
            score=score,
            method='my_detector'
        )
```

### Adding a Framework Adapter

```python
from sentinelml.adapters.base import BaseAdapter

class MyFrameworkAdapter(BaseAdapter):
    def __init__(self, model, **kwargs):
        self.model = model
        self.sentinel = kwargs.get('sentinel')

    def predict(self, X, **kwargs):
        # Pre-check with Sentinel
        if self.sentinel:
            check = self.sentinel.assess_batch(X)
            if any(c.trust_score < 0.5 for c in check):
                warnings.warn("Low trust predictions detected")

        # Original prediction
        return self.model.predict(X)

    def predict_proba(self, X, **kwargs):
        # Add uncertainty from Sentinel
        base_proba = self.model.predict_proba(X)
        if self.sentinel:
            trust = [self.sentinel.assess(x).trust_score for x in X]
            # Adjust probabilities based on trust
            return self._adjust_proba(base_proba, trust)
        return base_proba
```

---

## Deployment Patterns

### 1. Embedded Mode (In-Process)

```python
# Application integrates Sentinel directly
from sentinelml import Sentinel

sentinel = Sentinel().fit(X_train)

# In prediction pipeline
def predict_with_safety(model, x):
    check = sentinel.assess(x)
    if not check.is_trustworthy:
        return {"error": "Low trust", "fallback": "human_review"}
    return model.predict(x)
```

**Pros:** Low latency, simple
**Cons:** Tight coupling, scaling limitations

### 2. Sidecar Mode (Separate Process)

```
┌─────────────┐      HTTP/gRPC      ┌─────────────┐
│   ML App    │ ◄─────────────────► │  Sentinel   │
│  (Docker)   │    /assess calls    │  Server     │
└─────────────┘                     └─────────────┘
```

```python
# Client side
from sentinelml.infrastructure.serving import SentinelClient

client = SentinelClient("http://localhost:8000")
result = client.assess(x)
```

**Pros:** Independent scaling, language agnostic
**Cons:** Network overhead, deployment complexity

### 3. Streaming Mode (Kafka)

```
┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌─────────┐
│  App    │───►│  Kafka  │───►│  Sentinel   │───►│  Alert  │
│ Events  │    │  Topic  │    │  Consumer   │    │  Store  │
└─────────┘    └─────────┘    └─────────────┘    └─────────┘
```

```python
# From sentinelml.infrastructure.streaming
from sentinelml.infrastructure.streaming import KafkaConsumer

consumer = KafkaConsumer(
    topic="model-predictions",
    sentinel=sentinel,
    alert_handler=lambda r: send_alert(r) if not r.is_trustworthy else None
)
consumer.start()
```

### 4. Serverless Mode (AWS Lambda, etc.)

```python
# lambda_function.py
from sentinelml import Sentinel

sentinel = None  # Global for cold start reuse

def lambda_handler(event, context):
    global sentinel
    if sentinel is None:
        sentinel = load_sentinel_from_s3()

    x = parse_event(event)
    result = sentinel.assess(x)
    return format_response(result)
```

---

## Performance Considerations

### Computational Complexity

| Component | Training | Inference | Notes |
|-----------|----------|-----------|-------|
| KS Drift | O(n) | O(1) | Per-feature |
| MMD Drift | O(n²) | O(n) | Use approximation for large n |
| Mahalanobis | O(n) | O(d²) | d = n_features |
| Isolation Forest | O(nt log n) | O(nt) | t = n_trees |
| HNSW | O(n log n) | O(log n) | Approximate NN |
| MC Dropout | - | O(m) | m = forward passes |

### Memory Requirements

```python
# Estimate memory usage
sentinel = Sentinel(
    drift_detector=MMDDriftDetector(),  # Stores reference kernel matrix
    trust_model=MahalanobisTrust(),       # Stores covariance matrix
    familiarity=HNSWFamiliarity()         # Stores graph index
)

# For n=10k samples, d=100 features:
# - MMD: ~800MB (kernel matrix)
# - Mahalanobis: ~80KB (covariance)
# - HNSW: ~40MB (index)
```

### Optimization Strategies

1. **Subsampling**: Use representative subset for large reference data
2. **Dimensionality Reduction**: PCA before drift detection
3. **Caching**: Store pre-computed statistics
4. **Approximation**: HNSW instead of exact KD-Tree
5. **Batched I/O**: Process samples in chunks

```python
# Optimized configuration for large-scale
sentinel = Sentinel(
    drift_detector=MMDDriftDetector(
        n_permutations=100,  # Reduce from default 1000
        kernel_approximation='random_fourier'  # Use RBF approximation
    ),
    trust_model=MahalanobisTrust(
        assume_centered=True,  # Skip mean computation if pre-centered
        store_precision=False  # Don't store inverse covariance
    ),
    familiarity=HNSWFamiliarity(
        M=16,  # Reduce connections
        ef_construction=100  # Reduce accuracy for speed
    )
)
```

---

## Configuration Schema

### Full Configuration Example

```yaml
# sentinel-config.yaml
version: "2.0"

sentinel:
  # Core settings
  ensemble_strategy: "adaptive"  # weighted, voting, adaptive
  verbose: true
  random_state: 42

  # Drift detection
  drift_detector:
    type: "mmd"
    threshold: 0.05
    n_permutations: 1000
    kernel: "rbf"
    gamma: "scale"

  # Trust scoring
  trust_model:
    type: "mahalanobis"
    assume_centered: false
    store_precision: true

  # OOD detection
  familiarity:
    type: "hnsw"
    k: 5
    M: 16
    ef_construction: 200
    ef_search: 50
    metric: "euclidean"

# Domain-specific extensions
genai:
  guardrails:
    input:
      - type: "prompt_injection"
        threshold: 0.7
        model: "sentinelml/prompt-guard"
      - type: "pii_detection"
        entities: ["email", "phone", "ssn", "credit_card"]

    output:
      - type: "hallucination_detection"
        method: "self_consistency"
        n_samples: 5
        threshold: 0.5

    uncertainty:
      type: "semantic_entropy"
      n_samples: 10
      aggregation: "mean"

rag:
  retrieval:
    relevance_threshold: 0.7
    diversity_check: true
    max_contexts: 5

  generation:
    faithfulness_check: true
    citation_verify: true
    answer_relevance: true

  end_to_end:
    metrics: ["faithfulness", "answer_relevancy", "context_recall"]
    evaluator: "ragas"

# Infrastructure
infrastructure:
  serving:
    backend: "fastapi"  # or "grpc"
    host: "0.0.0.0"
    port: 8000
    workers: 4

  storage:
    vector_store: "faiss"  # or "chroma", "pinecone"
    checkpoint_store: "redis"

  streaming:
    enabled: true
    backend: "kafka"
    brokers: ["localhost:9092"]
    consumer_group: "sentinel-ml"
```

---

## Migration Guide (v1.0 → v2.0)

### Breaking Changes

| v1.0 | v2.0 | Migration |
|------|------|-----------|
| `from sentinelml.core import Sentinel` | `from sentinelml import Sentinel` | Update import |
| `sentinel.assess(x)["trust"]` | `sentinel.assess(x).trust_score` | Use attribute access |
| `sentinel.assess(x)["drift"]` | `sentinel.assess(x).has_drift` | Use attribute access |
| `BenchmarkComparison.evaluate()` | `BenchmarkComparison.evaluate_anomaly_detection()` | New API |
| `viz.plot_trust()` | `viz.plot_trust_dashboard()` | Updated signature |

### Compatibility Layer

```python
# Old code still works with compatibility layer
from sentinelml.benchmarks import LegacyBenchmarkComparison

benchmark = LegacyBenchmarkComparison(sentinel, model)
scores = benchmark.evaluate(X, y)  # Returns dict as before
```

---

## Appendix: Class Hierarchy

```
sentinelml/
├── core/
│   ├── base.py
│   │   └── BaseComponent (ABC)
│   │       ├── BaseDriftDetector
│   │       ├── BaseTrustModel
│   │       ├── BaseFamiliarityModel
│   │       ├── BaseUncertainty
│   │       ├── BaseGuardrail
│   │       └── BaseAdapter
│   │
│   ├── sentinel.py
│   │   └── Sentinel (orchestrator)
│   │
│   ├── pipeline.py
│   │   └── SentinelPipeline (sequential)
│   │
│   ├── ensemble.py
│   │   └── AdaptiveTrustEnsemble (weighted combo)
│   │
│   └── report.py
│       ├── TrustReport
│       ├── DriftReport
│       ├── GuardrailReport
│       └── RAGReport
│
└── [domains]/
    └── [implementations of BaseComponent]
```

---

**Last Updated**: 2024-03-10
**Version**: 2.0.0
**Maintainers**: SentinelML Team
