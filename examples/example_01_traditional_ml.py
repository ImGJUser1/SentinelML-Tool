"""
Example 1: Traditional ML Monitoring with SentinelML

This example demonstrates:
- Drift detection on tabular data
- Trust scoring for anomaly detection
- Batch processing for production
- Visualization of results

Use case: Monitoring a customer churn prediction model
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sentinelml import KDTreeFamiliarity, MahalanobisTrust, MMDDriftDetector, Sentinel
from sentinelml.viz import plot_trust_dashboard


def generate_customer_data(n_samples=1000, drift=False):
    """Generate synthetic customer churn data."""
    np.random.seed(42 if not drift else 123)

    if drift:
        # Simulate data drift - changed feature distribution
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=5,
            n_redundant=3,
            n_classes=2,
            class_sep=0.5,  # Reduced separation (harder problem)
            flip_y=0.1,  # More noise
            random_state=123,
        )
        # Add systematic shift
        X += np.random.normal(2, 1, X.shape)
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=5,
            n_redundant=3,
            n_classes=2,
            class_sep=1.0,
            flip_y=0.05,
            random_state=42,
        )

    feature_names = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "support_calls",
        "satisfaction_score",
        "contract_length",
        "payment_delay",
        "feature_8",
        "feature_9",
        "feature_10",
    ]

    df = pd.DataFrame(X, columns=feature_names)
    df["churn"] = y
    return df


def main():
    print("=" * 60)
    print("Traditional ML Monitoring Example")
    print("=" * 60)

    # 1. Generate reference (training) data
    print("\n1. Generating reference data (training period)...")
    reference_data = generate_customer_data(n_samples=1000, drift=False)
    X_ref = reference_data.drop("churn", axis=1).values
    y_ref = reference_data["churn"].values

    # 2. Train model
    print("2. Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_ref, y_ref)
    print(f"   Training accuracy: {model.score(X_ref, y_ref):.3f}")

    # 3. Initialize Sentinel with multiple monitoring strategies
    print("\n3. Initializing Sentinel monitoring system...")
    sentinel = Sentinel(
        drift_detector=MMDDriftDetector(
            threshold=0.05, n_permutations=1000  # Significance level  # Statistical power
        ),
        trust_model=MahalanobisTrust(),
        familiarity_model=KDTreeFamiliarity(k=10),
        verbose=True,
    )

    # 4. Fit on reference data
    print("\n4. Fitting Sentinel on reference data...")
    sentinel.fit(X_ref)

    # 5. Test on normal production data (same distribution)
    print("\n5. Monitoring normal production data...")
    normal_data = generate_customer_data(n_samples=500, drift=False)
    X_normal = normal_data.drop("churn", axis=1).values

    normal_results = sentinel.assess_batch(X_normal, batch_size=100)
    normal_trust_scores = [r.trust_score for r in normal_results]
    normal_drift_detected = sum(1 for r in normal_results if r.has_drift)

    print(f"   Samples: {len(normal_results)}")
    print(f"   Mean trust score: {np.mean(normal_trust_scores):.3f}")
    print(
        f"   Drift detected: {normal_drift_detected}/{len(normal_results)} ({100*normal_drift_detected/len(normal_results):.1f}%)"
    )

    # 6. Test on drifted data (simulating data quality issues)
    print("\n6. Monitoring DRIFTED production data...")
    drift_data = generate_customer_data(n_samples=500, drift=True)
    X_drift = drift_data.drop("churn", axis=1).values

    drift_results = sentinel.assess_batch(X_drift, batch_size=100)
    drift_trust_scores = [r.trust_score for r in drift_results]
    drift_detected_count = sum(1 for r in drift_results if r.has_drift)

    print(f"   Samples: {len(drift_results)}")
    print(f"   Mean trust score: {np.mean(drift_trust_scores):.3f}")
    print(
        f"   Drift detected: {drift_detected_count}/{len(drift_results)} ({100*drift_detected_count/len(drift_results):.1f}%)"
    )

    # 7. Model performance comparison
    print("\n7. Model performance impact:")
    y_pred_normal = model.predict(X_normal)
    y_pred_drift = model.predict(X_drift)

    # Simulate ground truth (in reality, you'd have actual labels)
    y_normal_true = normal_data["churn"].values
    y_drift_true = drift_data["churn"].values

    from sklearn.metrics import accuracy_score

    acc_normal = accuracy_score(y_normal_true, y_pred_normal)
    acc_drift = accuracy_score(y_drift_true, y_pred_drift)

    print(f"   Normal data accuracy: {acc_normal:.3f}")
    print(f"   Drifted data accuracy: {acc_drift:.3f}")
    print(f"   Performance drop: {acc_normal - acc_drift:.3f}")

    # 8. Visualization
    print("\n8. Generating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Trust score distributions
    axes[0, 0].hist(normal_trust_scores, bins=30, alpha=0.6, label="Normal", color="green")
    axes[0, 0].hist(drift_trust_scores, bins=30, alpha=0.6, label="Drifted", color="red")
    axes[0, 0].axvline(x=0.7, color="black", linestyle="--", label="Threshold")
    axes[0, 0].set_xlabel("Trust Score")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Trust Score Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Trust scores over time
    axes[0, 1].plot(normal_trust_scores, "g-", alpha=0.7, label="Normal")
    axes[0, 1].plot(
        range(len(normal_trust_scores), len(normal_trust_scores) + len(drift_trust_scores)),
        drift_trust_scores,
        "r-",
        alpha=0.7,
        label="Drifted",
    )
    axes[0, 1].axhline(y=0.7, color="black", linestyle="--", label="Threshold")
    axes[0, 1].set_xlabel("Sample Index")
    axes[0, 1].set_ylabel("Trust Score")
    axes[0, 1].set_title("Trust Scores Over Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Drift detection rate
    categories = ["Normal Data", "Drifted Data"]
    drift_rates = [
        normal_drift_detected / len(normal_results),
        drift_detected_count / len(drift_results),
    ]
    colors = ["green", "red"]
    axes[1, 0].bar(categories, drift_rates, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel("Drift Detection Rate")
    axes[1, 0].set_title("Drift Detection Comparison")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Model performance
    axes[1, 1].bar(
        ["Normal", "Drifted"], [acc_normal, acc_drift], color=["green", "red"], alpha=0.7
    )
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_title("Model Performance Impact")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("traditional_ml_monitoring.png", dpi=150, bbox_inches="tight")
    print("   Saved: traditional_ml_monitoring.png")

    # 9. Alert generation
    print("\n9. Alert summary:")
    low_trust_normal = sum(1 for s in normal_trust_scores if s < 0.7)
    low_trust_drift = sum(1 for s in drift_trust_scores if s < 0.7)

    print(f"   Normal data - Low trust alerts: {low_trust_normal}/{len(normal_results)}")
    print(f"   Drifted data - Low trust alerts: {low_trust_drift}/{len(drift_results)}")

    if low_trust_drift > len(drift_results) * 0.3:
        print("   ⚠️  ALERT: Significant data drift detected! Recommend model retraining.")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
