"""
Direction-Aware Loss Functions for Financial Prediction
Combines price prediction with direction accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionAwareLoss(nn.Module):
    """
    Combined loss that penalizes both price errors and direction errors

    Loss = α * MSE(price) + β * DirectionLoss + γ * MagnitudeLoss

    Where:
    - MSE(price): Standard mean squared error for price prediction
    - DirectionLoss: Binary cross-entropy for direction (up/down)
    - MagnitudeLoss: Error in predicted magnitude of change
    """

    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2):
        """
        Args:
            alpha: Weight for price MSE loss
            beta: Weight for direction loss (most important for trading)
            gamma: Weight for magnitude loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_returns, true_returns, prev_prices=None):
        """
        Args:
            pred_returns: Predicted returns [batch_size] or [batch_size, 1]
            true_returns: Actual returns [batch_size] or [batch_size, 1]
            prev_prices: Previous prices for calculating direction [batch_size] (optional)

        Returns:
            Combined loss value
        """
        # Ensure tensors are 1D
        if pred_returns.dim() > 1:
            pred_returns = pred_returns.squeeze(-1)
        if true_returns.dim() > 1:
            true_returns = true_returns.squeeze(-1)

        # 1. MSE Loss on returns
        mse_loss = self.mse(pred_returns, true_returns)

        # 2. Direction Loss (most critical for trading)
        # Convert returns to direction: 1 if up, 0 if down
        true_direction = (true_returns > 0).float()
        pred_direction_logits = pred_returns  # Use raw predictions as logits

        # Binary cross-entropy for direction
        direction_loss = self.bce(pred_direction_logits, true_direction)

        # 3. Magnitude Loss (how much it moved)
        # Penalize errors in the magnitude of movement
        pred_magnitude = torch.abs(pred_returns)
        true_magnitude = torch.abs(true_returns)
        magnitude_loss = self.mse(pred_magnitude, true_magnitude)

        # Combined loss
        total_loss = (
            self.alpha * mse_loss + self.beta * direction_loss + self.gamma * magnitude_loss
        )

        return total_loss, {
            "mse_loss": mse_loss.item(),
            "direction_loss": direction_loss.item(),
            "magnitude_loss": magnitude_loss.item(),
            "total_loss": total_loss.item(),
        }


class DirectionAccuracyMetric:
    """Calculate direction accuracy for evaluation"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, pred_returns, true_returns):
        """
        Update metrics with batch predictions

        Args:
            pred_returns: Predicted returns [batch_size]
            true_returns: Actual returns [batch_size]
        """
        pred_direction = (pred_returns > 0).float()
        true_direction = (true_returns > 0).float()

        self.correct += (pred_direction == true_direction).sum().item()
        self.total += len(pred_returns)

    def compute(self):
        """Compute direction accuracy"""
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total


class BalancedDirectionLoss(nn.Module):
    """
    Direction loss with class balancing for up/down movements
    Handles imbalanced datasets (e.g., bull markets with mostly up movements)
    """

    def __init__(self, alpha=0.5, beta=0.5):
        """
        Args:
            alpha: Weight for price loss
            beta: Weight for direction loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, pred_returns, true_returns):
        """
        Args:
            pred_returns: Predicted returns [batch_size] or [batch_size, 1]
            true_returns: Actual returns [batch_size] or [batch_size, 1]

        Returns:
            Combined loss with balanced direction component
        """
        # Ensure tensors are 1D
        if pred_returns.dim() > 1:
            pred_returns = pred_returns.squeeze(-1)
        if true_returns.dim() > 1:
            true_returns = true_returns.squeeze(-1)

        # 1. MSE Loss
        mse_loss = self.mse(pred_returns, true_returns)

        # 2. Balanced Direction Loss
        true_direction = (true_returns > 0).float()

        # Calculate class weights to balance up/down
        n_up = true_direction.sum()
        n_down = (1 - true_direction).sum()
        total = len(true_direction)

        if n_up > 0 and n_down > 0:
            weight_up = total / (2 * n_up)
            weight_down = total / (2 * n_down)
        else:
            weight_up = weight_down = 1.0

        # Weighted BCE
        pred_direction_logits = pred_returns
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_direction_logits, true_direction, reduction="none"
        )

        # Apply class weights
        weights = torch.where(true_direction == 1, weight_up, weight_down)
        direction_loss = (bce_loss * weights).mean()

        # Combined loss
        total_loss = self.alpha * mse_loss + self.beta * direction_loss

        return total_loss, {
            "mse_loss": mse_loss.item(),
            "direction_loss": direction_loss.item(),
            "total_loss": total_loss.item(),
            "n_up": n_up.item(),
            "n_down": n_down.item(),
        }


def calculate_direction_metrics(pred_returns, true_returns):
    """
    Calculate comprehensive direction metrics

    Returns:
        dict with direction accuracy, precision, recall, F1
    """
    # Ensure tensors are 1D
    if pred_returns.dim() > 1:
        pred_returns = pred_returns.squeeze(-1)
    if true_returns.dim() > 1:
        true_returns = true_returns.squeeze(-1)

    pred_direction = (pred_returns > 0).float()
    true_direction = (true_returns > 0).float()

    # True positives, false positives, etc.
    tp = ((pred_direction == 1) & (true_direction == 1)).sum().item()
    fp = ((pred_direction == 1) & (true_direction == 0)).sum().item()
    tn = ((pred_direction == 0) & (true_direction == 0)).sum().item()
    fn = ((pred_direction == 0) & (true_direction == 1)).sum().item()

    # Metrics
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "direction_accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1_score": f1 * 100,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }
