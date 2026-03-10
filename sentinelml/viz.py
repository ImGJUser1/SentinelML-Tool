import time
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
"""
Visualization utilities for SentinelML.

Provides comprehensive plotting functions for:
- Trust scores and trends
- Drift detection results
- Model performance analysis
- RAG evaluation metrics
- Agent trajectory monitoring
- Uncertainty quantification
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# Type aliases for clarity
Step = Dict[str, Any]
MetricsDict = Dict[str, List[float]]
Figure = Any  # Could be more specific if needed

# Optional matplotlib import
try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Optional plotly for interactive plots
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib required for visualization. " "Install: pip install matplotlib seaborn"
        )


def plot_trust(
    trust_scores: npt.ArrayLike,
    timestamps: Optional[npt.ArrayLike] = None,
    threshold: float = 0.7,
    title: str = "Trust Score Over Time",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Figure]:
    """
    Plot trust scores over time with threshold and confidence bands.

    Parameters
    ----------
    trust_scores : array-like
        Trust scores in [0, 1].
    timestamps : array-like, optional
        Timestamps or indices for x-axis.
    threshold : float, default=0.7
        Trustworthiness threshold line.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object if show=False.
    """
    _check_matplotlib()

    if len(trust_scores) == 0:
        raise ValueError("No trust scores provided to plot")

    trust_scores = np.asarray(trust_scores)
    if timestamps is None:
        timestamps = np.arange(len(trust_scores))
    else:
        timestamps = np.asarray(timestamps)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot trust scores
    ax.plot(timestamps, trust_scores, "b-", linewidth=2, label="Trust Score", alpha=0.8)

    # Add threshold line
    ax.axhline(threshold, color="r", linestyle="--", linewidth=2, label=f"Threshold ({threshold})")

    # Color regions based on trust level
    ax.fill_between(timestamps, 0, threshold, alpha=0.1, color="red", label="Low Trust Zone")
    ax.fill_between(timestamps, threshold, 1, alpha=0.1, color="green", label="High Trust Zone")

    # Highlight violations
    violations = trust_scores < threshold
    if np.any(violations):
        ax.scatter(
            timestamps[violations],
            trust_scores[violations],
            color="red",
            s=50,
            zorder=5,
            label="Violations",
        )

    # Rolling average
    if len(trust_scores) > 10:
        window = min(50, len(trust_scores) // 10)
        rolling = np.convolve(trust_scores, np.ones(window) / window, mode="valid")
        ax.plot(
            timestamps[window - 1 :],
            rolling,
            "g--",
            linewidth=2,
            alpha=0.7,
            label=f"Moving Avg (window={window})",
        )

    ax.set_xlabel("Sample / Time", fontsize=12)
    ax.set_ylabel("Trust Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig


def plot_drift(
    p_values: npt.ArrayLike,
    drift_detected: Optional[npt.ArrayLike] = None,
    threshold: float = 0.05,
    title: str = "Drift Detection (p-values)",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Figure]:
    """
    Plot drift detection p-values with significance threshold.

    Parameters
    ----------
    p_values : array-like
        P-values from drift detection.
    drift_detected : array-like, optional
        Boolean array indicating detected drift.
    threshold : float, default=0.05
        Significance threshold.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object if show=False.
    """
    _check_matplotlib()

    if len(p_values) == 0:
        raise ValueError("No p-values provided to plot")

    p_values = np.asarray(p_values)
    if drift_detected is None:
        drift_detected = p_values < threshold

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(p_values))

    # Plot p-values
    colors = ["red" if d else "blue" for d in drift_detected]
    ax.scatter(x, p_values, c=colors, s=30, alpha=0.6, zorder=3)
    ax.plot(x, p_values, "b-", alpha=0.3, linewidth=1)

    # Threshold line
    ax.axhline(
        threshold,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Significance Threshold (α={threshold})",
    )

    # Shade drift regions
    drift_regions = _find_contiguous_regions(drift_detected)
    for start, end in drift_regions:
        ax.axvspan(start, end, alpha=0.2, color="red", label="Drift Detected" if start == 0 else "")

    ax.set_xlabel("Sample", fontsize=12)
    ax.set_ylabel("p-value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Add statistics text
    n_drift = np.sum(drift_detected)
    stats_text = f"Drift detected: {n_drift}/{len(p_values)} ({100 * n_drift / len(p_values):.1f}%)"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig


def plot_trust_dashboard(
    reports: List[Any],
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Figure]:
    """
    Create comprehensive trust dashboard from TrustReport objects.

    Parameters
    ----------
    reports : list of TrustReport
        List of trust reports over time.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object if show=False.
    """
    _check_matplotlib()

    if len(reports) == 0:
        raise ValueError("No reports provided to plot")

    # Extract data from reports
    trust_scores = np.array([r.trust_score for r in reports])
    confidences = np.array([r.confidence for r in reports])
    drift_flags = np.array([r.has_drift for r in reports])
    violation_flags = np.array([r.has_violations for r in reports])
    timestamps = np.arange(len(reports))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Trust score over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(timestamps, trust_scores, "b-", linewidth=2, label="Trust Score")
    ax1.fill_between(timestamps, 0, 0.7, alpha=0.1, color="red")
    ax1.fill_between(timestamps, 0.7, 1, alpha=0.1, color="green")
    ax1.axhline(0.7, color="r", linestyle="--", alpha=0.7)
    ax1.set_ylabel("Trust Score")
    ax1.set_title("Trust Score Over Time", fontweight="bold")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # 2. Confidence over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(timestamps, confidences, "g-", linewidth=2)
    ax2.fill_between(timestamps, confidences, alpha=0.3, color="green")
    ax2.set_ylabel("Confidence")
    ax2.set_title("Model Confidence", fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # 3. Drift detection
    ax3 = fig.add_subplot(gs[1, 1])
    colors = ["red" if d else "blue" for d in drift_flags]
    ax3.scatter(timestamps, drift_flags.astype(int), c=colors, s=20)
    ax3.set_ylabel("Drift Detected")
    ax3.set_title("Drift Events", fontweight="bold")
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["No", "Yes"])
    ax3.grid(True, alpha=0.3)

    # 4. Component scores heatmap
    ax4 = fig.add_subplot(gs[2, 0])
    if hasattr(reports[0], "raw_scores") and reports[0].raw_scores:
        score_names = list(reports[0].raw_scores.keys())
        score_matrix = np.array([[r.raw_scores.get(k, 0) for k in score_names] for r in reports])
        im = ax4.imshow(score_matrix.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax4.set_yticks(range(len(score_names)))
        ax4.set_yticklabels(score_names)
        ax4.set_xlabel("Sample")
        ax4.set_title("Component Scores", fontweight="bold")
        plt.colorbar(im, ax=ax4)

    # 5. Violations histogram
    ax5 = fig.add_subplot(gs[2, 1])
    violation_counts = [
        len(r.guardrail_reports) if hasattr(r, "guardrail_reports") else 0 for r in reports
    ]
    ax5.hist(violation_counts, bins=20, color="orange", edgecolor="black", alpha=0.7)
    ax5.set_xlabel("Number of Violations")
    ax5.set_ylabel("Frequency")
    ax5.set_title("Guardrail Violations Distribution", fontweight="bold")
    ax5.grid(True, alpha=0.3)

    plt.suptitle("SentinelML Trust Dashboard", fontsize=16, fontweight="bold", y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig


def plot_uncertainty_distribution(
    uncertainties: npt.ArrayLike,
    predictions: Optional[npt.ArrayLike] = None,
    bins: int = 50,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Figure]:
    """
    Plot distribution of uncertainty estimates.

    Parameters
    ----------
    uncertainties : array-like
        Uncertainty values.
    predictions : array-like, optional
        Model predictions for coloring.
    bins : int
        Number of histogram bins.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object if show=False.
    """
    _check_matplotlib()

    if len(uncertainties) == 0:
        raise ValueError("No uncertainty values provided to plot")

    uncertainties = np.asarray(uncertainties)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1 = axes[0]
    ax1.hist(uncertainties, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)
    ax1.axvline(
        np.mean(uncertainties),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(uncertainties):.3f}",
    )
    ax1.axvline(
        np.median(uncertainties),
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(uncertainties):.3f}",
    )
    ax1.set_xlabel("Uncertainty")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Uncertainty Distribution", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot with predictions
    ax2 = axes[1]
    if predictions is not None:
        predictions = np.asarray(predictions)
        unique_preds = np.unique(predictions)
        data = [uncertainties[predictions == p] for p in unique_preds]
        bp = ax2.boxplot(data, labels=[f"Class {p}" for p in unique_preds], patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
    else:
        ax2.boxplot(uncertainties, patch_artist=True)
        ax2.set_xticklabels(["All Samples"])

    ax2.set_ylabel("Uncertainty")
    ax2.set_title("Uncertainty by Prediction", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig


def plot_rag_metrics(
    metrics: Dict[str, List[float]],
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Figure]:
    """
    Plot RAG evaluation metrics over queries.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric names to lists of values.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object if show=False.
    """
    _check_matplotlib()

    if not metrics:
        raise ValueError("No metrics provided to plot")

    # Check all metric lists have same length
    lengths = [len(v) for v in metrics.values()]
    if len(set(lengths)) > 1:
        raise ValueError("All metric lists must have the same length")

    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))

    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx]
        values = np.asarray(values)
        x = np.arange(len(values))

        # Line plot
        ax.plot(x, values, "o-", color=colors[idx], linewidth=2, markersize=4, alpha=0.7)

        # Mean line
        mean_val = np.mean(values)
        ax.axhline(mean_val, color="r", linestyle="--", alpha=0.7, label=f"Mean: {mean_val:.3f}")

        # Rolling average
        if len(values) > 10:
            window = min(10, len(values) // 5)
            rolling = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(
                x[window - 1 :], rolling, "--", color="orange", linewidth=2, label=f"MA({window})"
            )

        ax.set_xlabel("Query Index")
        ax.set_ylabel("Score")
        ax.set_title(metric_name.replace("_", " ").title(), fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("RAG Evaluation Metrics", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig


def plot_agent_trajectory(
    steps: List[Step],
    metrics: Optional[MetricsDict] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Figure]:
    """
    Plot agent trajectory showing steps, tools used, and reasoning path.

    Parameters
    ----------
    steps : list of dict
        List of agent steps with keys:
        - step: step number
        - tool: tool used (if any)
        - thought: reasoning text
        - observation: observation text
        - confidence: confidence score (optional)
    metrics : dict, optional
        Additional metrics over time.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object if show=False.
    """
    _check_matplotlib()

    if not steps:
        raise ValueError("No steps provided to plot")

    n_steps = len(steps)
    step_nums = np.arange(1, n_steps + 1)

    # Extract data
    tools = [s.get("tool", "none") for s in steps]
    confidences = [s.get("confidence", 0.5) for s in steps]

    # Unique tools for coloring
    unique_tools = list(set(tools))
    if not unique_tools:
        unique_tools = ["none"]

    tool_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tools)))
    tool_to_color = dict(zip(unique_tools, tool_colors))

    fig = plt.figure(figsize=figsize)

    if metrics:
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 2, 1])
    else:
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 2])

    # 1. Tool usage timeline
    ax1 = fig.add_subplot(gs[0, :])
    for i, (step, tool) in enumerate(zip(step_nums, tools)):
        ax1.scatter(step, 1, color=tool_to_color[tool], s=200, marker="|", linewidth=3, alpha=0.7)

    ax1.set_xlim(0.5, n_steps + 0.5)
    ax1.set_ylim(0.9, 1.1)
    ax1.set_yticks([])
    ax1.set_xlabel("Step")
    ax1.set_title("Tool Usage Timeline", fontweight="bold")

    # Create legend for tools
    legend_elements = [
        mpatches.Patch(color=color, label=tool, alpha=0.7) for tool, color in tool_to_color.items()
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # 2. Confidence trajectory
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(step_nums, confidences, "b-o", linewidth=2, markersize=6, alpha=0.7)
    ax2.fill_between(step_nums, 0, confidences, alpha=0.3, color="blue")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Confidence")
    ax2.set_title("Confidence Over Steps", fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # 3. Step type distribution
    ax3 = fig.add_subplot(gs[1, 1])
    tool_counts = {tool: tools.count(tool) for tool in unique_tools}
    ax3.bar(
        tool_counts.keys(),
        tool_counts.values(),
        color=[tool_to_color[t] for t in tool_counts.keys()],
        alpha=0.7,
    )
    ax3.set_xlabel("Tool Type")
    ax3.set_ylabel("Count")
    ax3.set_title("Step Type Distribution", fontweight="bold")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3)

    # 4. Additional metrics if provided
    if metrics:
        ax4 = fig.add_subplot(gs[2, :])
        for metric_name, values in metrics.items():
            ax4.plot(
                step_nums[: len(values)],
                values,
                "o-",
                linewidth=2,
                markersize=4,
                label=metric_name,
                alpha=0.7,
            )
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Score")
        ax4.set_title("Additional Metrics", fontweight="bold")
        ax4.legend(loc="best", fontsize=8)
        ax4.grid(True, alpha=0.3)

    # Add step descriptions as text
    thought_text = "\n".join(
        [f"Step {i+1}: {s.get('thought', '')[:50]}..." for i, s in enumerate(steps[:3])]
    )
    if len(steps) > 3:
        thought_text += f"\n... and {len(steps) - 3} more steps"

    fig.text(
        0.02,
        0.02,
        thought_text,
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
    )

    plt.suptitle("Agent Trajectory Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig


def plot_component_importance(
    importance_scores: Dict[str, float],
    top_n: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Figure]:
    """
    Plot component importance scores as horizontal bar chart.

    Parameters
    ----------
    importance_scores : dict
        Dictionary mapping component names to importance scores.
    top_n : int, optional
        Show only top N components.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object if show=False.
    """
    _check_matplotlib()

    if not importance_scores:
        raise ValueError("No importance scores provided to plot")

    # Sort by importance
    sorted_items = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

    if top_n:
        sorted_items = sorted_items[:top_n]

    names = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar chart
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(range(len(names)), scores, color=colors, alpha=0.8)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            ha="left",
            va="center",
            fontsize=10,
        )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Component Importance Analysis", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()  # Highest importance at top
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig


def plot_correlation_matrix(
    data: npt.ArrayLike,
    feature_names: Optional[List[str]] = None,
    title: str = "Feature Correlation Matrix",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Figure]:
    """
    Plot correlation matrix of features.

    Parameters
    ----------
    data : array-like
        Feature data matrix.
    feature_names : list of str, optional
        Names of features.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object if show=False.
    """
    _check_matplotlib()

    data = np.asarray(data)
    if data.size == 0:
        raise ValueError("No data provided to plot")

    corr_matrix = np.corrcoef(data.T)

    fig, ax = plt.subplots(figsize=figsize)

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot heatmap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # Add correlation values
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if not mask[i, j]:
                ax.text(
                    j,
                    i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black" if abs(corr_matrix[i, j]) < 0.5 else "white",
                    fontsize=8,
                )

    # Set ticks
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(data.shape[1])]

    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(feature_names, fontsize=10)

    # Colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig


def create_interactive_dashboard(
    trust_scores: npt.ArrayLike,
    drift_pvalues: Optional[npt.ArrayLike] = None,
    component_scores: Optional[Dict[str, List[float]]] = None,
    title: str = "SentinelML Interactive Dashboard",
    trust_threshold: float = 0.7,
    drift_threshold: float = 0.05,
) -> Optional[Any]:
    """
    Create interactive dashboard using Plotly.

    Parameters
    ----------
    trust_scores : array-like
        Trust scores over time.
    drift_pvalues : array-like, optional
        Drift detection p-values.
    component_scores : dict, optional
        Component scores over time.
    title : str
        Dashboard title.
    trust_threshold : float, default=0.7
        Threshold for trust scores.
    drift_threshold : float, default=0.05
        Threshold for drift p-values.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive dashboard figure.
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly required for interactive dashboard. " "Install: pip install plotly"
        )

    if len(trust_scores) == 0:
        raise ValueError("No trust scores provided to plot")

    trust_scores = np.asarray(trust_scores)
    timestamps = np.arange(len(trust_scores))

    # Create subplots
    if component_scores:
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Trust Score",
                "Drift Detection",
                "Component Scores",
                "Score Distribution",
                "Rolling Statistics",
                "Violation Summary",
            ),
            specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None]],
        )
    else:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Trust Score",
                "Drift Detection",
                "Score Distribution",
                "Rolling Statistics",
            ),
        )

    # 1. Trust score line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=trust_scores,
            mode="lines+markers",
            name="Trust Score",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    # Add threshold line
    fig.add_hline(
        y=trust_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({trust_threshold})",
        row=1,
        col=1,
    )

    # 2. Drift p-values
    if drift_pvalues is not None:
        drift_pvalues = np.asarray(drift_pvalues)
        colors = ["red" if p < drift_threshold else "blue" for p in drift_pvalues]

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=drift_pvalues,
                mode="lines+markers",
                marker=dict(color=colors, size=8),
                line=dict(color="gray", width=1, dash="dot"),
                name="p-values",
            ),
            row=1,
            col=2,
        )

        fig.add_hline(
            y=drift_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"α={drift_threshold}",
            row=1,
            col=2,
        )

    # 3. Component scores
    if component_scores:
        for component_name, scores in component_scores.items():
            fig.add_trace(
                go.Scatter(
                    x=timestamps[: len(scores)],
                    y=scores,
                    mode="lines",
                    name=component_name,
                    line=dict(width=2),
                ),
                row=2,
                col=1,
            )

    # 4. Score distribution
    fig.add_trace(
        go.Histogram(
            x=trust_scores, nbinsx=30, name="Distribution", marker_color="lightblue", opacity=0.7
        ),
        row=2,
        col=2,
    )

    # 5. Rolling statistics
    window = min(10, max(1, len(trust_scores) // 10))
    rolling_mean = np.convolve(trust_scores, np.ones(window) / window, mode="valid")
    rolling_std = np.array(
        [np.std(trust_scores[max(0, i - window) : i + 1]) for i in range(len(trust_scores))]
    )

    target_row = 3 if component_scores else 2
    target_col = 1 if component_scores else 2

    fig.add_trace(
        go.Scatter(
            x=timestamps[window - 1 :],
            y=rolling_mean,
            mode="lines",
            name=f"MA({window})",
            line=dict(color="orange", width=2),
        ),
        row=target_row,
        col=target_col,
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=trust_scores + rolling_std,
            mode="lines",
            name="+1 Std",
            line=dict(color="gray", width=1, dash="dot"),
            showlegend=False,
        ),
        row=target_row,
        col=target_col,
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=trust_scores - rolling_std,
            mode="lines",
            name="-1 Std",
            line=dict(color="gray", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.2)",
            showlegend=False,
        ),
        row=target_row,
        col=target_col,
    )

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        height=800 if component_scores else 600,
        showlegend=True,
        template="plotly_white",
    )

    # Update axes
    fig.update_xaxes(title_text="Sample", row=1, col=1)
    fig.update_xaxes(title_text="Sample", row=1, col=2)
    fig.update_xaxes(title_text="Sample", row=2, col=1)
    fig.update_xaxes(title_text="Trust Score", row=2, col=2)
    if component_scores:
        fig.update_xaxes(title_text="Sample", row=3, col=1)

    fig.update_yaxes(title_text="Trust Score", row=1, col=1, range=[0, 1])
    fig.update_yaxes(title_text="p-value", row=1, col=2, range=[0, 1])
    fig.update_yaxes(title_text="Score", row=2, col=1, range=[0, 1])
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    if component_scores:
        fig.update_yaxes(title_text="Trust Score", row=3, col=1, range=[0, 1])

    return fig


def _find_contiguous_regions(boolean_array: npt.ArrayLike) -> List[Tuple[int, int]]:
    """
    Find contiguous regions where boolean array is True.

    Parameters
    ----------
    boolean_array : array-like
        Boolean array.

    Returns
    -------
    list of tuples
        List of (start, end) indices for True regions.
    """
    boolean_array = np.asarray(boolean_array)
    regions = []

    in_region = False
    start = 0

    for i, val in enumerate(boolean_array):
        if val and not in_region:
            start = i
            in_region = True
        elif not val and in_region:
            regions.append((start, i - 1))
            in_region = False

    if in_region:
        regions.append((start, len(boolean_array) - 1))

    return regions


__all__ = [
    "plot_trust",
    "plot_drift",
    "plot_trust_dashboard",
    "plot_uncertainty_distribution",
    "plot_rag_metrics",
    "plot_agent_trajectory",
    "plot_component_importance",
    "plot_correlation_matrix",
    "create_interactive_dashboard",
    "HAS_MATPLOTLIB",
    "HAS_PLOTLY",
]
