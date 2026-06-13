# Loss Functions

Meridian.AI uses two loss functions: **BalancedDirectionLoss** for training and **DirectionAwareLoss** as an alternative. Both are implemented in `meridianalgo/direction_loss.py`.

---

## BalancedDirectionLoss (used in training)

```python
pred_scaled = pred_return * 100   # percent units
true_scaled = true_return * 100

loss = 0.6 * HuberLoss(pred_scaled, true_scaled)
     + 0.4 * BCE(pred_scaled, direction)
```

This is the primary training objective. Both components operate in **percent units** (`return_scale=100`). It jointly optimises two objectives:

### Component 1: Huber regression (60%)

```python
SmoothL1Loss(pred_scaled, true_scaled)
```

Huber loss is quadratic for small errors (`|error| < 1`) and linear for large errors, so it is robust to outlier returns. The percent scaling is what makes it work: raw daily returns (~0.005) are ~200× too small for SmoothL1's quadratic region, so on raw returns the regression gradient was negligible and training bottomed out at a degenerate "predict ~0" equilibrium (loss ≈ 0.6·0 + 0.4·log 2 ≈ 0.277) with uncalibrated magnitudes. In percent units a typical error is ~0.5 — squarely in the quadratic region.

### Component 2: Direction BCE (40%)

The percent-scaled prediction is used directly as the direction logit:

```python
pred_logits = pred_scaled                       # percent units ARE the logits
bce = F.binary_cross_entropy_with_logits(pred_logits, direction)
```

A ~0.5% predicted move gives `sigmoid(0.5) ≈ 0.62` up-probability — well-ranged logits with a meaningful gradient. At the optimum the logit approaches `logit(P(up | x))`, so the model can express the market's base rate.

**Class weighting is opt-in and off by default** (`balance_classes=False`). With weighting on, the BCE optimum for a weak-signal input is a 50/50 logit instead of the true base rate — it erases the market's up-drift prior, the one signal the always-up baseline gets for free. A v7 development run with weighting on collapsed to a constant slightly-bearish prediction (48.7% vs 52.0% always-up). Enable it only for genuinely balanced research datasets.

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
