# Loss Functions

Meridian.AI uses two loss functions: **BalancedDirectionLoss** for training and **DirectionAwareLoss** as an alternative. Both are implemented in `meridianalgo/direction_loss.py`.

---

## BalancedDirectionLoss (used in training)

```python
loss = 0.6 * HuberLoss(pred_return, true_return)
     + 0.4 * WeightedBCE(pred_return, direction)
```

This is the primary training objective. It jointly optimises two objectives:

### Component 1: Huber regression (60%)

```python
SmoothL1Loss(pred_returns, true_returns)
```

Huber loss is quadratic for small errors (`|error| < 1`) and linear for large errors. This makes it more robust than MSE to outlier returns that survive the ±100%/±20% clip threshold — a single extreme return won't dominate the loss surface.

### Component 2: Weighted direction BCE (40%)

Financial datasets tend to be directionally imbalanced — during a bull market, most daily returns are positive. Predicting "up" for every sample would score ~60% accuracy while learning nothing.

The fix: class-weighted BCE.

```python
# Compute class weights from this batch
n_up   = (true_returns > 0).sum()
n_down = (true_returns <= 0).sum()
N      = len(true_returns)

weight_up   = N / (2 * n_up)    # upsamples minority class
weight_down = N / (2 * n_down)

# Scale raw predictions to logit range (returns are ~0.001 scale)
pred_logits = pred_returns * 10.0

# Weighted binary cross-entropy
bce = F.binary_cross_entropy_with_logits(pred_logits, direction, reduction='none')
loss = (bce * weights).mean()
```

The `* 10.0` logit scaling is necessary because raw return predictions are on the order of 0.001–0.01. Feeding these directly into BCE produces outputs near `log(2)` regardless of the prediction — the direction signal would be invisible. Scaling to `[-1, 1]` or wider makes the BCE gradient meaningful.

### Why 60/40 split?

Price regression provides the gradient signal that shapes the overall loss landscape. Direction is the metric we care about for trading, but it's binary — its BCE gradient is high variance. A 60/40 split lets regression dominate the optimisation trajectory while direction acts as a steering correction.

---

## DirectionAwareLoss (alternative)

A three-component loss available as an alternative:

```python
loss = 0.4 * MSE(pred, true)
     + 0.4 * BCE(pred, direction)
     + 0.2 * MSE(|pred|, |true|)   # magnitude term
```

The magnitude term (`gamma=0.2`) penalises errors in how much the price moves, not just direction. This matters when the model is correct on direction but the magnitude is wildly off — a ±5% predicted move vs a 0.1% actual move is a very different position-sizing error.

Use `DirectionAwareLoss` when magnitude calibration matters more than raw direction accuracy. `BalancedDirectionLoss` is preferred for training since it's more stable (Huber vs MSE).

---

## Direction metrics

After each validation epoch, `calculate_direction_metrics()` computes:

```python
pred_direction = (pred_returns > 0).float()   # 1 = predicted up
true_direction = (true_returns > 0).float()   # 1 = actually up
```

| Metric | Formula | Meaning |
|--------|---------|---------|
| `direction_accuracy` | `(TP + TN) / N × 100` | Overall % correct direction |
| `precision` | `TP / (TP + FP) × 100` | When we predict up, how often are we right? |
| `recall` | `TP / (TP + FN) × 100` | Of all actual up days, how many did we catch? |
| `f1_score` | `2 × P × R / (P + R)` | Harmonic mean of precision and recall |

All values are in percent (0–100). The checkpoint saves `direction_accuracy` in `metadata`.

**Threshold**: A model that can't beat 50% direction accuracy has not learned anything beyond a coin flip. The `test_checkpoint_health.py` test enforces `direction_accuracy >= 50.0`.

---

## DirectionAccuracyMetric

A stateful metric accumulator for streaming evaluation:

```python
metric = DirectionAccuracyMetric()
metric.reset()

for batch in val_loader:
    pred, _ = model(x)
    metric.update(pred.squeeze(), y)

acc = metric.compute()   # returns float, 0–100
```

Use this when you want to accumulate accuracy across many batches without keeping all predictions in memory.

---

## See also

- [Training guide](TRAINING.md) — how losses are used in the training loop
- [Architecture](ARCHITECTURE.md) — model output that feeds into the loss
