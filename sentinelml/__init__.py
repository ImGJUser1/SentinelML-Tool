import re
"""
SentinelML: Unified Reliability Engine for AI/ML Systems.

A comprehensive framework for monitoring, evaluating, and ensuring the reliability
of machine learning systems across traditional ML, deep learning, generative AI,
RAG pipelines, and agentic systems.

Version 2.0.0 - Complete Rewrite with Modular Architecture
"""

__version__ = "2.0.0"
__author__ = "SentinelML Team"
__license__ = "MIT"

from sentinelml.agents.reasoning import LogicChecker, StepConsistency
from sentinelml.agents.state import BudgetManager, CheckpointManager

# Agent exports
from sentinelml.agents.trajectory import LoopDetector, StepValidator, ToolMonitor
from sentinelml.core.ensemble import AdaptiveTrustEnsemble
from sentinelml.core.pipeline import SentinelPipeline
from sentinelml.core.report import DriftReport, GuardrailReport, TrustReport

# Core exports - Primary API
from sentinelml.core.sentinel import Sentinel

# Deep Learning exports
from sentinelml.deep_learning.uncertainty import (
    DeepEnsembleUncertainty,
    EvidentialNetwork,
    MCDropoutUncertainty,
    TemperatureScaling,
)

# GenAI exports
from sentinelml.genai.guardrails.input import (
    IntentClassifier,
    PIIDetector,
    PromptInjectionDetector,
    ToxicityFilter,
)
from sentinelml.genai.guardrails.output import (
    CitationVerifier,
    ConsistencyCheck,
    HallucinationDetector,
    SchemaValidator,
)
from sentinelml.genai.uncertainty import LexicalSimilarity, SemanticEntropy, TokenLogProb
from sentinelml.rag.end_to_end import ARESEvaluator, LatencyTracker, RAGASEvaluator
from sentinelml.rag.generation import AnswerRelevance, CitationAccuracy, FaithfulnessChecker

# RAG exports
from sentinelml.rag.retrieval import CoverageAnalyzer, DiversityMetrics, RelevanceScorer

# Traditional ML exports
from sentinelml.traditional.drift import (
    AdversarialDriftDetector,
    KSDriftDetector,
    MMDDriftDetector,
    PSIDetector,
)
from sentinelml.traditional.familiarity import (
    HNSWFamiliarity,
    KDTreeFamiliarity,
    KernelDensityFamiliarity,
)
from sentinelml.traditional.trust import (
    ConformalPredictor,
    IsolationForestTrust,
    MahalanobisTrust,
    VAETrust,
)


# Utility warnings for optional dependencies
def _check_optional_deps():
    """Warn about missing optional dependencies."""
    import warnings

    optional_modules = {
        "torch": "PyTorch (for deep learning)",
        "tensorflow": "TensorFlow (for deep learning)",
        "transformers": "HuggingFace Transformers (for NLP)",
        "openai": "OpenAI (for LLM APIs)",
        "langchain": "LangChain (for RAG)",
    }

    for module, description in optional_modules.items():
        try:
            __import__(module)
        except ImportError:
            pass  # Silently skip, will error when actually used


# Run check on import
_check_optional_deps()

__all__ = [
    # Version
    "__version__",
    # Core
    "Sentinel",
    "SentinelPipeline",
    "AdaptiveTrustEnsemble",
    "TrustReport",
    "DriftReport",
    "GuardrailReport",
    # Traditional
    "KSDriftDetector",
    "PSIDetector",
    "MMDDriftDetector",
    "AdversarialDriftDetector",
    "MahalanobisTrust",
    "IsolationForestTrust",
    "VAETrust",
    "ConformalPredictor",
    "KDTreeFamiliarity",
    "HNSWFamiliarity",
    "KernelDensityFamiliarity",
    # Deep Learning
    "MCDropoutUncertainty",
    "DeepEnsembleUncertainty",
    "TemperatureScaling",
    "EvidentialNetwork",
    # GenAI
    "PromptInjectionDetector",
    "PIIDetector",
    "ToxicityFilter",
    "IntentClassifier",
    "HallucinationDetector",
    "ConsistencyCheck",
    "SchemaValidator",
    "CitationVerifier",
    "SemanticEntropy",
    "LexicalSimilarity",
    "TokenLogProb",
    # RAG
    "RelevanceScorer",
    "CoverageAnalyzer",
    "DiversityMetrics",
    "FaithfulnessChecker",
    "AnswerRelevance",
    "CitationAccuracy",
    "RAGASEvaluator",
    "ARESEvaluator",
    "LatencyTracker",
    # Agents
    "StepValidator",
    "ToolMonitor",
    "LoopDetector",
    "LogicChecker",
    "StepConsistency",
    "BudgetManager",
    "CheckpointManager",
]
