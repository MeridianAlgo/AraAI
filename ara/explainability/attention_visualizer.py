"""
Attention Visualizer for Transformer Models
Extracts and visualizes attention weights for explainability
"""

import numpy as np
from typing import Dict, List, Any, Optional
import warnings

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Attention visualization will be limited.")

from ara.core.exceptions import ModelError, ValidationError


class AttentionVisualizer:
    """
    Visualizes attention weights from Transformer models
    Provides insights into which time steps the model focuses on
    """

    def __init__(self, model: Optional[Any] = None, layer_names: Optional[List[str]] = None):
        """
        Initialize attention visualizer

        Args:
            model: Transformer model (PyTorch)
            layer_names: Names of attention layers to extract
        """
        self.model = model
        self.layer_names = layer_names
        self.attention_weights = {}
        self.hooks = []

    def register_hooks(self, model: Optional[Any] = None):
        """
        Register forward hooks to capture attention weights

        Args:
            model: Model to register hooks on (uses self.model if None)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for attention visualization")

        if model is not None:
            self.model = model

        if self.model is None:
            raise ModelError("No model provided for hook registration")

        # Clear existing hooks
        self.remove_hooks()
        self.attention_weights = {}

        # Register hooks on attention layers
        def get_attention_hook(name):
            def hook(module, input, output):
                # Store attention weights
                # Format depends on the specific attention implementation
                if isinstance(output, tuple) and len(output) > 1:
                    # Some implementations return (output, attention_weights)
                    self.attention_weights[name] = output[1].detach().cpu().numpy()
                else:
                    # Try to extract from module attributes
                    if hasattr(module, "attention_weights"):
                        self.attention_weights[name] = (
                            module.attention_weights.detach().cpu().numpy()
                        )

            return hook

        # Find and register hooks on attention layers
        for name, module in self.model.named_modules():
            if "attention" in name.lower() or isinstance(module, nn.MultiheadAttention):
                if self.layer_names is None or name in self.layer_names:
                    hook = module.register_forward_hook(get_attention_hook(name))
                    self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_attention_weights(
        self, model: Any, input_data: np.ndarray, layer_idx: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract attention weights from a forward pass

        Args:
            model: Transformer model
            input_data: Input data (batch_size, seq_len, features)
            layer_idx: Specific layer index to extract (None for all)

        Returns:
            Dict mapping layer names to attention weight arrays
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for attention extraction")

        # Register hooks if not already done
        if not self.hooks:
            self.register_hooks(model)

        # Convert input to tensor
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.FloatTensor(input_data)
        else:
            input_tensor = input_data

        # Forward pass to capture attention
        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)

        # Filter by layer index if specified
        if layer_idx is not None:
            filtered_weights = {}
            for name, weights in self.attention_weights.items():
                if f"layer.{layer_idx}" in name or f"layers.{layer_idx}" in name:
                    filtered_weights[name] = weights
            return filtered_weights

        return self.attention_weights.copy()

    def get_attention_summary(
        self, attention_weights: np.ndarray, time_steps: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for attention weights

        Args:
            attention_weights: Attention weight matrix (n_heads, seq_len, seq_len)
                              or (batch, n_heads, seq_len, seq_len)
            time_steps: Optional labels for time steps

        Returns:
            Dict with attention statistics
        """
        # Handle different input shapes
        if attention_weights.ndim == 4:
            # (batch, n_heads, seq_len, seq_len) - average over batch
            attention_weights = attention_weights.mean(axis=0)

        if attention_weights.ndim != 3:
            raise ValidationError(
                "attention_weights must be 3D or 4D", {"shape": attention_weights.shape}
            )

        n_heads, seq_len, _ = attention_weights.shape

        # Average attention across heads
        avg_attention = attention_weights.mean(axis=0)

        # Find most attended positions for each query
        most_attended = np.argmax(avg_attention, axis=1)

        # Calculate attention entropy (measure of focus)
        epsilon = 1e-10
        attention_entropy = -np.sum(avg_attention * np.log(avg_attention + epsilon), axis=1)

        # Identify which positions receive most attention overall
        total_attention_received = avg_attention.sum(axis=0)

        summary = {
            "n_heads": n_heads,
            "seq_length": seq_len,
            "avg_attention": avg_attention.tolist(),
            "most_attended_positions": most_attended.tolist(),
            "attention_entropy": attention_entropy.tolist(),
            "total_attention_received": total_attention_received.tolist(),
            "max_attention_per_head": attention_weights.max(axis=(1, 2)).tolist(),
            "min_attention_per_head": attention_weights.min(axis=(1, 2)).tolist(),
        }

        if time_steps:
            summary["time_step_labels"] = time_steps

        return summary

    def get_top_attended_positions(
        self, attention_weights: np.ndarray, query_position: int, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get top K positions that a query attends to

        Args:
            attention_weights: Attention weights (n_heads, seq_len, seq_len)
            query_position: Position of the query
            top_k: Number of top positions to return

        Returns:
            List of dicts with position and attention score
        """
        if attention_weights.ndim == 4:
            attention_weights = attention_weights.mean(axis=0)

        # Average across heads
        avg_attention = attention_weights.mean(axis=0)

        # Get attention for this query position
        query_attention = avg_attention[query_position]

        # Get top K positions
        top_indices = np.argsort(query_attention)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "position": int(idx),
                    "attention_score": float(query_attention[idx]),
                    "percentage": float(query_attention[idx] * 100),
                }
            )

        return results

    def analyze_attention_patterns(self, attention_weights: np.ndarray) -> Dict[str, Any]:
        """
        Analyze attention patterns to identify interesting behaviors

        Args:
            attention_weights: Attention weights (n_heads, seq_len, seq_len)

        Returns:
            Dict with pattern analysis
        """
        if attention_weights.ndim == 4:
            attention_weights = attention_weights.mean(axis=0)

        n_heads, seq_len, _ = attention_weights.shape

        patterns = {"heads": []}

        for head_idx in range(n_heads):
            head_attention = attention_weights[head_idx]

            # Check for diagonal attention (local focus)
            diagonal_strength = np.trace(head_attention) / seq_len

            # Check for uniform attention (global focus)
            1.0 / seq_len
            uniformity = 1.0 - np.std(head_attention)

            # Check for position bias (attending to specific positions)
            position_bias = head_attention.mean(axis=0)
            max_bias_position = int(np.argmax(position_bias))

            # Check for recency bias (attending to recent positions)
            recency_weights = np.arange(seq_len, 0, -1) / seq_len
            recency_score = np.corrcoef(position_bias, recency_weights)[0, 1]

            head_pattern = {
                "head_idx": head_idx,
                "diagonal_strength": float(diagonal_strength),
                "uniformity": float(uniformity),
                "max_bias_position": max_bias_position,
                "max_bias_score": float(position_bias[max_bias_position]),
                "recency_score": float(recency_score),
                "pattern_type": self._classify_attention_pattern(
                    diagonal_strength, uniformity, recency_score
                ),
            }

            patterns["heads"].append(head_pattern)

        return patterns

    def _classify_attention_pattern(
        self, diagonal_strength: float, uniformity: float, recency_score: float
    ) -> str:
        """Classify the type of attention pattern"""
        if diagonal_strength > 0.5:
            return "local"
        elif uniformity > 0.8:
            return "global"
        elif recency_score > 0.5:
            return "recency-biased"
        elif recency_score < -0.5:
            return "distant-biased"
        else:
            return "mixed"

    def create_attention_heatmap_data(
        self,
        attention_weights: np.ndarray,
        head_idx: Optional[int] = None,
        time_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare attention data for heatmap visualization

        Args:
            attention_weights: Attention weights
            head_idx: Specific head to visualize (None for average)
            time_labels: Labels for time steps

        Returns:
            Dict with heatmap data
        """
        if attention_weights.ndim == 4:
            attention_weights = attention_weights.mean(axis=0)

        # Select specific head or average
        if head_idx is not None:
            if head_idx >= attention_weights.shape[0]:
                raise ValidationError(
                    f"head_idx {head_idx} out of range",
                    {"n_heads": attention_weights.shape[0]},
                )
            heatmap_data = attention_weights[head_idx]
        else:
            heatmap_data = attention_weights.mean(axis=0)

        seq_len = heatmap_data.shape[0]

        # Create labels if not provided
        if time_labels is None:
            time_labels = [f"t-{seq_len-i-1}" for i in range(seq_len)]

        return {
            "data": heatmap_data.tolist(),
            "x_labels": time_labels,
            "y_labels": time_labels,
            "title": (
                f"Attention Heatmap (Head {head_idx})"
                if head_idx is not None
                else "Average Attention Heatmap"
            ),
            "xlabel": "Key Position",
            "ylabel": "Query Position",
        }

    def get_temporal_attention_flow(self, attention_weights: np.ndarray) -> Dict[str, Any]:
        """
        Analyze how attention flows through time

        Args:
            attention_weights: Attention weights (n_heads, seq_len, seq_len)

        Returns:
            Dict with temporal flow analysis
        """
        if attention_weights.ndim == 4:
            attention_weights = attention_weights.mean(axis=0)

        # Average across heads
        avg_attention = attention_weights.mean(axis=0)
        seq_len = avg_attention.shape[0]

        # Calculate attention flow for each position
        flow_data = []
        for pos in range(seq_len):
            # How much this position attends to past vs future
            past_attention = avg_attention[pos, :pos].sum() if pos > 0 else 0
            current_attention = avg_attention[pos, pos]
            future_attention = avg_attention[pos, pos + 1 :].sum() if pos < seq_len - 1 else 0

            flow_data.append(
                {
                    "position": pos,
                    "past_attention": float(past_attention),
                    "current_attention": float(current_attention),
                    "future_attention": float(future_attention),
                    "total_attention": float(past_attention + current_attention + future_attention),
                }
            )

        return {
            "flow_by_position": flow_data,
            "overall_past_bias": float(np.mean([d["past_attention"] for d in flow_data])),
            "overall_current_bias": float(np.mean([d["current_attention"] for d in flow_data])),
            "overall_future_bias": float(np.mean([d["future_attention"] for d in flow_data])),
        }
