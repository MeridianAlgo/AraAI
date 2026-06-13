# Technical Indicators Reference

MeridianModel uses **44 technical indicators** computed from raw OHLCV data. All features are computed without lookahead — each row at time `t` only uses price/volume data up to and including timestep `t`.

**v7.0.0: every feature is scale-invariant** — a ratio, percentage, or bounded oscillator. The v6 set fed 14 raw price/volume *levels* (`sma_5` … `ema_200`, `bb_upper/lower`, `kc_upper/lower`, raw `macd`, raw `momentum`, `atr`, `volume_sma`) into a z-score scaler fitted **across symbols** whose prices range from 0.6 (EURGBP) to 190 (GBPJPY) — or $4 to $900 for stocks. After cross-symbol normalisation those columns mostly encoded *which symbol* a window came from, not anything predictive. Ratios (`close/sma − 1`, `macd/close`, …) carry the same information in a symbol-independent scale.

After extraction, all features are z-score normalised (train-split statistics) and clamped to `[-5, 5]` before being fed to the model.

---

## Feature list by index

| Index | Name | Formula / Description | Category |
|-------|------|-----------------------|----------|
| 0 | `returns` | `(close[t] - close[t-1]) / close[t-1]` | Price |
| 1 | `log_returns` | `ln(close[t] / close[t-1])` | Price |
| 2 | `volatility` | Rolling 20-day std of returns | Price |
| 3 | `hl_range_pct` | `(high - low) / close` — intraday range as % of price | Price |
| 4 | `close_vs_sma_5` | `close / SMA(5) - 1` | Trend |
| 5 | `close_vs_ema_5` | `close / EMA(5) - 1` | Trend |
| 6 | `close_vs_sma_10` | `close / SMA(10) - 1` | Trend |
| 7 | `close_vs_ema_10` | `close / EMA(10) - 1` | Trend |
| 8 | `close_vs_sma_20` | `close / SMA(20) - 1` | Trend |
| 9 | `close_vs_ema_20` | `close / EMA(20) - 1` | Trend |
| 10 | `close_vs_sma_50` | `close / SMA(50) - 1` | Trend |
| 11 | `close_vs_ema_50` | `close / EMA(50) - 1` | Trend |
| 12 | `close_vs_sma_200` | `close / SMA(200) - 1` | Trend |
| 13 | `close_vs_ema_200` | `close / EMA(200) - 1` | Trend |
| 14 | `rsi` | Relative Strength Index (14 periods) | Momentum |
| 15 | `rsi_fast` | RSI (7 periods) | Momentum |
| 16 | `stoch_rsi` | Stochastic RSI (14-period min/max of RSI) | Momentum |
| 17 | `macd` | `(EMA12 - EMA26) / close` — price-normalised MACD | Oscillator |
| 18 | `macd_signal` | EMA9 of the normalised MACD | Oscillator |
| 19 | `macd_hist` | `macd - macd_signal` | Oscillator |
| 20 | `bb_upper_dist` | `bb_upper / close - 1` (20-period SMA + 2σ band) | Volatility |
| 21 | `bb_lower_dist` | `bb_lower / close - 1` (20-period SMA - 2σ band) | Volatility |
| 22 | `bb_width` | `(bb_upper - bb_lower) / sma_20` | Volatility |
| 23 | `bb_pct` | `(close - bb_lower) / (bb_upper - bb_lower)` | Volatility |
| 24 | `volume_trend` | `SMA20(volume) / SMA100(volume) - 1` | Volume |
| 25 | `volume_ratio` | `volume / SMA20(volume)` | Volume |
| 26 | `obv_norm` | `(OBV - SMA20(OBV)) / std20(OBV)` — z-scored On-Balance Volume | Volume |
| 27 | `ret_5d` | `close[t] / close[t-5] - 1` — 5-day return | Momentum |
| 28 | `roc` | `(close[t] / close[t-10] - 1) × 100` — Rate of Change | Momentum |
| 29 | `williams_r` | Williams %R (14 periods) | Momentum |
| 30 | `stoch_k` | Stochastic %K (14 periods) | Oscillator |
| 31 | `stoch_d` | Stochastic %D (3-period SMA of %K) | Oscillator |
| 32 | `cci` | Commodity Channel Index (20 periods) | Oscillator |
| 33 | `kc_upper_dist` | `kc_upper / close - 1` (EMA20 + 2 × ATR) | Volatility |
| 34 | `kc_lower_dist` | `kc_lower / close - 1` (EMA20 - 2 × ATR) | Volatility |
| 35 | `kc_pct` | `(close - kc_lower) / (kc_upper - kc_lower)` | Volatility |
| 36 | `adx` | Average Directional Index (14 periods) | Trend Strength |
| 37 | `plus_di` | +DI directional indicator (14 periods) | Trend Strength |
| 38 | `minus_di` | -DI directional indicator (14 periods) | Trend Strength |
| 39 | `vol_regime` | `std20(returns) / std100(returns) - 1` — volatility regime | Volatility |
| 40 | `ret_20d` | `close[t] / close[t-20] - 1` — monthly momentum | Momentum |
| 41 | `atr_pct` | `ATR(14) / close` — volatility as % of price | Trend Strength |
| 42 | `zscore_20` | `(close - sma_20) / std_20` | Mean Reversion |
| 43 | `dist_from_high` | `(close - high_252) / high_252` — distance from 52-week high | Mean Reversion |

---

## Category descriptions

### Price (indices 0–3)

Short-term dynamics without indicator lag: daily return, log return (additive, closer to normal), 20-day realised volatility, and the intraday range as a percentage of price (replaces the raw-price ATR feature; ATR itself survives as `atr_pct`).

### Trend — moving-average distances (indices 4–13)

Instead of raw SMA/EMA levels, the model sees the *distance* of the close from each average (`close / MA - 1`) at five timescales. Same trend information, but a +2% stretch above the 50-day average reads identically for a $4 stock and a $900 stock — or a 0.6 and a 190 forex pair. Having both SMA and EMA distance at each period lets the model learn their divergence.

### Momentum (indices 14–16, 27–29, 40)

RSI at two speeds plus Stochastic RSI; 5-day, 10-day (ROC) and 20-day returns; Williams %R for where the close sits in the recent range.

### Oscillators (indices 17–19, 30–32)

MACD family normalised by price so it is comparable across symbols; Stochastic %K/%D; CCI (already self-normalising via MAD).

### Volatility (indices 20–23, 33–35, 39)

Bollinger and Keltner band *distances* as a percentage of price (replaces raw band levels), band width, %B / %KC position within the bands, and a volatility-regime ratio (20-day vs 100-day realised vol) that tells the model whether the market is heating up or cooling down.

### Volume (indices 24–26)

Volume trend (20-day vs 100-day average), volume spike ratio vs its 20-day average, and z-scored OBV (replaces the unbounded raw OBV cumsum).

### Trend Strength (indices 36–38, 41)

ADX with +DI/−DI (trend strength and direction), and ATR as a percentage of price.

### Mean Reversion (indices 42–43)

20-day z-score of price and drawdown from the 52-week high.

---

## Why no lookahead?

Every feature at row `t` is computed from data up to and including `t` (trailing rolling windows only). The target is the `t → t+1` close-to-close return.

**Forex caveat:** the daily `*=X` source bars are internally inconsistent — day-t high/low span a later window than the stored close, so the *bar itself* leaks t+1 information even though the features are trailing. Forex therefore trains and evaluates with a **1-day embargo** (`embargo_days=1`): the input window ends at `t-1` when predicting the `t → t+1` return. See `scripts/diag_feat_corr.py` and the v1.2.0 changelog entry.

---

## Normalisation

After all 44 features are extracted for a training run:

1. **Z-score**: Subtract train-split mean, divide by train-split std (never fitted on the validation split). Mean and std are stored in the checkpoint.
2. **Clamp**: `torch.clamp(x, -5, 5)`. Prevents near-constant features from blowing up to ±100σ after normalisation and saturating activations.

At inference time, the same `scaler_mean` and `scaler_std` from the checkpoint are applied before feeding the model.

---

## See also

- [Architecture](ARCHITECTURE.md) — how the model processes these 44 features
- [Training guide](TRAINING.md) — data pipeline and normalisation details
