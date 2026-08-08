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

    Uses properly scaled logits via a learnable scaling factor instead of
    raw return predictions (which are near-zero and break BCE).
    """

    def __init__(self, alpha=0.6, beta=0.4, return_scale=100.0, balance_classes=False):
        """
        Args:
            alpha: Weight for price regression loss (primary signal)
            beta: Weight for direction loss (auxiliary signal)
            balance_classes: Reweight BCE so up/down contribute equally.
                Default False: with weighting on, the BCE optimum for a
                weak-signal input is logit 0 (50/50) instead of the true
                base rate logit(P(up|x)) — it erases the market's up-drift
                prior, the one signal the always-up baseline gets for free.
                A v7 holdout run with weighting on collapsed to a constant
                slightly-bearish prediction (48.7% vs 52.0% always-up).
            return_scale: Both losses operate on returns * return_scale
                (i.e. percent units). Raw daily returns (~0.005) are ~200x
                too small for SmoothL1's quadratic region, so the regression
                gradient was negligible and the optimum was the 0.277 "dead
                prediction" equilibrium (0.6 * ~0 + 0.4 * log(2)). It also
                let the direction term inflate |pred| unchecked — live
                checkpoints predicted ~17% daily moves with no regression
                pushback. In percent units a typical error is ~0.5, squarely
                in Huber's quadratic region, and the same scaling gives the
                direction term well-ranged logits (sigmoid(0.5) ~ 0.62).
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.return_scale = return_scale
        self.balance_classes = balance_classes
        self.huber = nn.SmoothL1Loss()

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

        # Work in percent units (see __init__ docstring). Predictions stay in
        # raw return scale everywhere outside the loss.
        pred_scaled = pred_returns * self.return_scale
        true_scaled = true_returns * self.return_scale

        # 1. Huber Loss (robust to outliers, better than MSE for financial data)
        regression_loss = self.huber(pred_scaled, true_scaled)

        # 2. Balanced Direction Loss — the percent-scaled prediction IS the logit
        true_direction = (true_returns > 0).float()
        pred_direction_logits = pred_scaled

        n_up = true_direction.sum()
        n_down = (1 - true_direction).sum()

        if self.balance_classes and n_up > 0 and n_down > 0:
            total = len(true_direction)
            weight_up = total / (2 * n_up)
            weight_down = total / (2 * n_down)
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_direction_logits, true_direction, reduction="none"
            )
            weights = torch.where(true_direction == 1, weight_up, weight_down)
            direction_loss = (bce_loss * weights).mean()
        else:
            direction_loss = F.binary_cross_entropy_with_logits(
                pred_direction_logits, true_direction
            )

        # Combined loss
        total_loss = self.alpha * regression_loss + self.beta * direction_loss

        return total_loss, {
            "regression_loss": regression_loss.item(),
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
