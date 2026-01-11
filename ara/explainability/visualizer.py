"""
Visualization Module for Explainability
Creates visual explanations including bar charts and heatmaps
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization features will be limited.")



class ExplainabilityVisualizer:
    """
    Creates visualizations for model explanations
    Generates bar charts, heatmaps, and other visual explanations
    """

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize visualizer

        Args:
            style: Matplotlib style to use
        """
        self.style = style
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(style)
            except:
                # Fallback to default style
                pass

    def plot_feature_contributions(
        self,
        top_features: List[Dict[str, Any]],
        title: str = "Top Feature Contributions",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create bar chart of feature contributions

        Args:
            top_features: List of feature dicts with contributions
            title: Chart title
            figsize: Figure size (width, height)
            save_path: Path to save figure (optional)

        Returns:
            Figure object or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            warnings.warn("Matplotlib not available for visualization")
            return None

        # Extract data
        feature_names = [f["feature_name"] for f in top_features]
        contributions = [f["contribution"] for f in top_features]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create bar chart
        colors = ["green" if c > 0 else "red" for c in contributions]
        bars = ax.barh(feature_names, contributions, color=colors, alpha=0.7)

        # Customize
        ax.set_xlabel("Contribution", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, (bar, contrib) in enumerate(zip(bars, contributions)):
            width = bar.get_width()
            label_x = width + (0.01 if width > 0 else -0.01)
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{contrib:.4f}",
                ha="left" if width > 0 else "right",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {save_path}")

        return fig

    def plot_contribution_percentages(
        self,
        top_features: List[Dict[str, Any]],
        title: str = "Feature Contribution Percentages",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create horizontal bar chart of contribution percentages

        Args:
            top_features: List of feature dicts with contribution_percentage
            title: Chart title
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Figure object or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        # Extract data
        feature_names = [f["feature_name"] for f in top_features]
        percentages = [f.get("contribution_percentage", 0) for f in top_features]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create bar chart
        bars = ax.barh(feature_names, percentages, color="steelblue", alpha=0.7)

        # Customize
        ax.set_xlabel("Contribution (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            width = bar.get_width()
            ax.text(
                width + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                ha="left",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_shap_waterfall(
        self,
        base_value: float,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: List[str],
        max_display: int = 10,
        title: str = "SHAP Waterfall Plot",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create waterfall plot showing how features contribute to prediction

        Args:
            base_value: Base/expected value
            shap_values: SHAP values for features
            feature_values: Actual feature values
            feature_names: Names of features
            max_display: Maximum features to display
            title: Chart title
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Figure object or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        # Sort by absolute SHAP value
        indices = np.argsort(np.abs(shap_values))[-max_display:][::-1]

        # Get top features
        top_shap = shap_values[indices]
        top_names = [feature_names[i] for i in indices]
        top_values = feature_values[indices]

        # Calculate cumulative values
        cumulative = np.cumsum(np.concatenate([[base_value], top_shap]))

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot bars
        y_pos = np.arange(len(top_names) + 2)

        # Base value
        ax.barh(0, base_value, color="gray", alpha=0.5, label="Base value")

        # Feature contributions
        for i, (name, shap_val, feat_val) in enumerate(
            zip(top_names, top_shap, top_values)
        ):
            color = "green" if shap_val > 0 else "red"
            ax.barh(i + 1, shap_val, left=cumulative[i], color=color, alpha=0.7)

            # Add feature value annotation
            label = f"{name}\n= {feat_val:.3f}"
            ax.text(-0.1, i + 1, label, ha="right", va="center", fontsize=8)

        # Final prediction
        final_pred = cumulative[-1]
        ax.barh(len(top_names) + 1, 0, left=final_pred, color="blue", alpha=0.5)
        ax.text(
            final_pred,
            len(top_names) + 1,
            f"  {final_pred:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(["Base"] + [""] * len(top_names) + ["Prediction"])
        ax.set_xlabel("Value", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        title: str = "Attention Heatmap",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create heatmap of attention weights

        Args:
            attention_weights: 2D array of attention weights
            x_labels: Labels for x-axis (key positions)
            y_labels: Labels for y-axis (query positions)
            title: Chart title
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Figure object or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        im = ax.imshow(attention_weights, cmap="YlOrRd", aspect="auto")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Weight", rotation=270, labelpad=20)

        # Set labels
        if x_labels:
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right")

        if y_labels:
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels)

        ax.set_xlabel("Key Position", fontsize=12)
        ax.set_ylabel("Query Position", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_feature_importance_ranking(
        self,
        feature_importance: Dict[str, float],
        top_n: int = 20,
        title: str = "Feature Importance Ranking",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create bar chart of feature importance ranking

        Args:
            feature_importance: Dict mapping features to importance scores
            top_n: Number of top features to display
            title: Chart title
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Figure object or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        # Sort and get top N
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
        )[:top_n]

        feature_names = [f[0] for f in sorted_features]
        importance_scores = [f[1] for f in sorted_features]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
        bars = ax.barh(feature_names, importance_scores, color=colors, alpha=0.8)

        # Customize
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for bar, score in zip(bars, importance_scores):
            width = bar.get_width()
            ax.text(
                width + max(importance_scores) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}",
                ha="left",
                va="center",
                fontsize=8,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_contribution_comparison(
        self,
        contributions1: np.ndarray,
        contributions2: np.ndarray,
        feature_names: List[str],
        label1: str = "Scenario 1",
        label2: str = "Scenario 2",
        top_n: int = 10,
        title: str = "Contribution Comparison",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create side-by-side comparison of contributions

        Args:
            contributions1: First set of contributions
            contributions2: Second set of contributions
            feature_names: Names of features
            label1: Label for first scenario
            label2: Label for second scenario
            top_n: Number of top features to display
            title: Chart title
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Figure object or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        # Get top features by combined importance
        combined_importance = np.abs(contributions1) + np.abs(contributions2)
        top_indices = np.argsort(combined_importance)[-top_n:][::-1]

        top_names = [feature_names[i] for i in top_indices]
        top_contrib1 = contributions1[top_indices]
        top_contrib2 = contributions2[top_indices]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

        # Plot first scenario
        colors1 = ["green" if c > 0 else "red" for c in top_contrib1]
        ax1.barh(top_names, top_contrib1, color=colors1, alpha=0.7)
        ax1.set_xlabel("Contribution", fontsize=11)
        ax1.set_title(label1, fontsize=12, fontweight="bold")
        ax1.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        ax1.grid(axis="x", alpha=0.3)

        # Plot second scenario
        colors2 = ["green" if c > 0 else "red" for c in top_contrib2]
        ax2.barh(top_names, top_contrib2, color=colors2, alpha=0.7)
        ax2.set_xlabel("Contribution", fontsize=11)
        ax2.set_title(label2, fontsize=12, fontweight="bold")
        ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        ax2.grid(axis="x", alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def close_all(self):
        """Close all matplotlib figures"""
        if MATPLOTLIB_AVAILABLE:
            plt.close("all")
