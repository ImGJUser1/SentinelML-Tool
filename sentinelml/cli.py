import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
"""
Command-line interface for SentinelML v2.0.

Provides commands for:
- Scanning datasets for drift and anomalies
- Evaluating model reliability
- Running monitoring server
- Generating reports
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sentinelml import __version__
from sentinelml.core.sentinel import Sentinel
from sentinelml.traditional.drift import KSDriftDetector, MMDDriftDetector
from sentinelml.traditional.familiarity import KDTreeFamiliarity
from sentinelml.traditional.trust import MahalanobisTrust


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="sentinelml",
        description=f"SentinelML v{__version__} - Unified Reliability Engine for AI/ML Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sentinelml scan data.csv --drift-detector mmd
  sentinelml evaluate model.pkl test.csv --output report.json
  sentinelml serve --port 8000 --config sentinel.yaml
        """,
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Scan command
    scan_parser = subparsers.add_parser(
        "scan", help="Scan dataset for drift, anomalies, and trust issues"
    )
    scan_parser.add_argument("file", type=str, help="Path to CSV or Parquet file")
    scan_parser.add_argument(
        "--reference", type=str, help="Path to reference dataset (if different from file)"
    )
    scan_parser.add_argument(
        "--drift-detector",
        choices=["ks", "mmd", "psi", "adversarial"],
        default="mmd",
        help="Drift detection method",
    )
    scan_parser.add_argument(
        "--trust-model",
        choices=["mahalanobis", "isolation_forest", "vae"],
        default="mahalanobis",
        help="Trust scoring method",
    )
    scan_parser.add_argument("--threshold", type=float, default=0.7, help="Trust score threshold")
    scan_parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    scan_parser.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size for processing"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model reliability on test data")
    eval_parser.add_argument("model", type=str, help="Path to model file (pickle or joblib)")
    eval_parser.add_argument("data", type=str, help="Path to test data (CSV or Parquet)")
    eval_parser.add_argument(
        "--labels", type=str, help='Column name for labels (if not "target" or "y")'
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default="evaluation_report.json",
        help="Output file for evaluation report",
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start monitoring server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")
    serve_parser.add_argument("--config", type=str, help="Path to configuration file (YAML)")
    serve_parser.add_argument(
        "--protocol", choices=["http", "grpc"], default="http", help="Server protocol"
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Generate configuration template")
    config_parser.add_argument(
        "--output", type=str, default="sentinel.yaml", help="Output file for configuration"
    )
    config_parser.add_argument(
        "--type",
        choices=["traditional", "deep", "genai", "rag", "agent"],
        default="traditional",
        help="Configuration type",
    )

    return parser


def cmd_scan(args: argparse.Namespace) -> int:
    """Execute scan command."""
    print(f"SentinelML v{__version__} - Scanning {args.file}")

    # Load data
    try:
        if args.file.endswith(".csv"):
            df = pd.read_csv(args.file)
        elif args.file.endswith(".parquet"):
            df = pd.read_parquet(args.file)
        else:
            print(f"Error: Unsupported file format: {args.file}")
            return 1
    except Exception as e:
        print(f"Error loading file: {e}")
        return 1

    # Extract features
    X = df.select_dtypes(include=[np.number]).values

    if len(X) == 0:
        print("Error: No numeric columns found in data")
        return 1

    print(f"Loaded {len(X)} samples with {X.shape[1]} features")

    # Determine reference data
    if args.reference:
        try:
            ref_df = (
                pd.read_csv(args.reference)
                if args.reference.endswith(".csv")
                else pd.read_parquet(args.reference)
            )
            X_ref = ref_df.select_dtypes(include=[np.number]).values
        except Exception as e:
            print(f"Error loading reference: {e}")
            return 1
    else:
        # Use first 20% as reference
        split_idx = len(X) // 5
        X_ref = X[:split_idx]
        X = X[split_idx:]
        print(f"Using first {len(X_ref)} samples as reference, {len(X)} for testing")

    # Create drift detector
    if args.drift_detector == "ks":
        drift_detector = KSDriftDetector(threshold=0.05)
    elif args.drift_detector == "mmd":
        drift_detector = MMDDriftDetector(threshold=0.05)
    else:
        print(f"Detector {args.drift_detector} not yet implemented in CLI")
        drift_detector = None

    # Create trust model
    if args.trust_model == "mahalanobis":
        trust_model = MahalanobisTrust()
    else:
        print(f"Trust model {args.trust_model} not yet implemented in CLI")
        trust_model = None

    # Create familiarity model
    familiarity = KDTreeFamiliarity()

    # Build Sentinel
    sentinel = Sentinel(drift_detector=drift_detector, trust_model=trust_model, verbose=True)

    print("\nFitting on reference data...")
    sentinel.fit(X_ref)

    print(f"\nAssessing {len(X)} samples...")

    # Process in batches
    results = []
    for i in range(0, len(X), args.batch_size):
        batch = X[i : i + args.batch_size]
        batch_results = [sentinel.assess(x, sample_id=f"sample_{i+j}") for j, x in enumerate(batch)]
        results.extend(batch_results)

        if (i // args.batch_size) % 10 == 0:
            print(f"  Processed {min(i + args.batch_size, len(X))}/{len(X)}")

    # Summarize results
    trust_scores = [r.trust_score for r in results]
    drift_detected = sum(1 for r in results if r.has_drift)
    untrustworthy = sum(1 for r in results if not r.is_trustworthy)

    print(f"\n{'='*50}")
    print("SCAN RESULTS")
    print(f"{'='*50}")
    print(f"Samples analyzed:     {len(results)}")
    print(f"Drift detected:       {drift_detected} ({100*drift_detected/len(results):.1f}%)")
    print(f"Untrustworthy:        {untrustworthy} ({100*untrustworthy/len(results):.1f}%)")
    print(f"Mean trust score:     {np.mean(trust_scores):.3f}")
    print(f"Min trust score:      {np.min(trust_scores):.3f}")
    print(f"Max trust score:      {np.max(trust_scores):.3f}")

    # Save results if requested
    if args.output:
        output_data = {
            "version": __version__,
            "command": "scan",
            "file": args.file,
            "summary": {
                "n_samples": len(results),
                "drift_detected": drift_detected,
                "untrustworthy": untrustworthy,
                "mean_trust": float(np.mean(trust_scores)),
                "min_trust": float(np.min(trust_scores)),
                "max_trust": float(np.max(trust_scores)),
            },
            "results": [r.to_dict() for r in results],
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Execute evaluate command."""
    print(f"SentinelML v{__version__} - Evaluating {args.model}")

    try:
        import joblib

        model = joblib.load(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Load data
    try:
        if args.data.endswith(".csv"):
            df = pd.read_csv(args.data)
        else:
            df = pd.read_parquet(args.data)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Extract features and labels
    label_col = (
        args.labels or "target" if "target" in df.columns else "y" if "y" in df.columns else None
    )

    if label_col and label_col in df.columns:
        y = df[label_col].values
        X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values
    else:
        print("Warning: No labels found, running unsupervised evaluation")
        y = None
        X = df.select_dtypes(include=[np.number]).values

    print(f"Loaded {len(X)} samples")

    # Generate predictions if labels available
    if y is not None and hasattr(model, "predict"):
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"Model accuracy: {accuracy:.3f}")

    # TODO: Full evaluation with Sentinel
    print("Full evaluation report generation - TODO in v2.0")

    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Execute serve command."""
    print(f"SentinelML v{__version__} - Starting server on {args.host}:{args.port}")

    if args.protocol == "grpc":
        print("gRPC server - TODO in v2.0")
        return 1
    else:
        try:
            from sentinelml.core.sentinel import Sentinel
            from sentinelml.infrastructure.serving.fastapi_server import FastAPIServer

            # Create default sentinel
            sentinel = Sentinel(verbose=True)

            server = FastAPIServer(sentinel=sentinel, host=args.host, port=args.port, verbose=True)
            server.fit()
            server.start(blocking=True)

        except ImportError as e:
            print(f"Error: {e}")
            print("Install serving dependencies: pip install sentinelml[serving]")
            return 1
        except Exception as e:
            print(f"Server error: {e}")
            return 1

    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Execute config command."""
    templates = {
        "traditional": """# SentinelML Configuration - Traditional ML
sentinel:
  drift_detector:
    type: mmd
    threshold: 0.05
    window_size: 1000

  trust_model:
    type: mahalanobis
    calibration: isotonic

  familiarity:
    type: kdtree
    k: 5

monitoring:
  batch_size: 1000
  check_interval: 3600
""",
        "deep": """# SentinelML Configuration - Deep Learning
sentinel:
  uncertainty:
    type: mc_dropout
    n_samples: 50

  feature_drift:
    layers: ['layer3', 'layer4']
    method: mmd

  adversarial:
    enabled: true
    epsilon: 0.03

monitoring:
  batch_size: 32
  device: auto
""",
        "genai": """# SentinelML Configuration - GenAI/LLM
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
      - type: schema_validation

  uncertainty:
    type: semantic_entropy
    n_samples: 10

llm:
  model: gpt-4
  temperature: 0.7
""",
        "rag": """# SentinelML Configuration - RAG
sentinel:
  retrieval:
    relevance_threshold: 0.7
    diversity_check: true

  generation:
    faithfulness_check: true
    citation_verify: true

  end_to_end:
    metrics: [faithfulness, answer_relevancy, context_recall]

vector_store:
  backend: faiss
  dimension: 384
""",
        "agent": """# SentinelML Configuration - Agents
sentinel:
  trajectory:
    step_validation: true
    loop_detection: true
    max_steps: 50

  tools:
    allowed: [search, calculator, code_executor]
    rate_limits:
      search: 60
      code_executor: 10

  budget:
    max_tokens: 10000
    max_cost: 1.00
""",
    }

    config = templates.get(args.type, templates["traditional"])

    with open(args.output, "w") as f:
        f.write(config)

    print(f"Configuration template written to: {args.output}")
    return 0


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handlers
    commands = {
        "scan": cmd_scan,
        "evaluate": cmd_evaluate,
        "serve": cmd_serve,
        "config": cmd_config,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)

    print(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
