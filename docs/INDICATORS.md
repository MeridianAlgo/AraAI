# Technical Indicators Reference

MeridianModel uses **44 technical indicators** computed from raw OHLCV data. All features are computed without lookahead — each row at time `t` only uses price/volume data up to and including timestep `t`.

After extraction, all features are z-score normalised and clamped to `[-10, 10]` before being fed to the model.

---

## Feature list by index

| Index | Name | Formula / Description | Category |
|-------|------|-----------------------|----------|
| 0 | `return` | `(close[t] - close[t-1]) / close[t-1]` | Price |
| 1 | `log_return` | `ln(close[t] / close[t-1])` | Price |
| 2 | `volatility_5` | Rolling 5-day std of returns | Price |
| 3 | `atr` | Average True Range (14 periods) | Price |
| 4 | `sma_5` | Simple Moving Average, 5 periods | Trend |
| 5 | `sma_10` | Simple Moving Average, 10 periods | Trend |
| 6 | `sma_20` | Simple Moving Average, 20 periods | Trend |
| 7 | `sma_50` | Simple Moving Average, 50 periods | Trend |
| 8 | `sma_200` | Simple Moving Average, 200 periods | Trend |
| 9 | `ema_5` | Exponential Moving Average, 5 periods | Trend |
| 10 | `ema_10` | Exponential Moving Average, 10 periods | Trend |
| 11 | `ema_20` | Exponential Moving Average, 20 periods | Trend |
| 12 | `ema_50` | Exponential Moving Average, 50 periods | Trend |
| 13 | `ema_200` | Exponential Moving Average, 200 periods | Trend |
| 14 | `rsi` | Relative Strength Index (14 periods) | Momentum |
| 15 | `rsi_fast` | RSI (7 periods) | Momentum |
| 16 | `stoch_rsi` | Stochastic RSI | Momentum |
| 17 | `momentum` | `close[t] - close[t-10]` | Momentum |
| 18 | `roc` | Rate of Change (10 periods): `(close[t] / close[t-10] - 1) × 100` | Momentum |
| 19 | `williams_r` | Williams %R (14 periods) | Momentum |
| 20 | `macd` | MACD line (EMA12 - EMA26) | Oscillator |
| 21 | `macd_signal` | MACD signal line (EMA9 of MACD) | Oscillator |
| 22 | `macd_hist` | MACD histogram (MACD - signal) | Oscillator |
| 23 | `stoch_k` | Stochastic %K (14 periods) | Oscillator |
| 24 | `stoch_d` | Stochastic %D (3-period SMA of %K) | Oscillator |
| 25 | `cci` | Commodity Channel Index (20 periods) | Oscillator |
| 26 | `bb_upper` | Bollinger Band upper (20-period SMA + 2σ) | Volatility |
| 27 | `bb_lower` | Bollinger Band lower (20-period SMA - 2σ) | Volatility |
| 28 | `bb_width` | `(bb_upper - bb_lower) / sma_20` | Volatility |
| 29 | `bb_pct` | `(close - bb_lower) / (bb_upper - bb_lower)` | Volatility |
| 30 | `kc_upper` | Keltner Channel upper (EMA20 + 1.5 × ATR) | Volatility |
| 31 | `kc_lower` | Keltner Channel lower (EMA20 - 1.5 × ATR) | Volatility |
| 32 | `kc_pct` | `(close - kc_lower) / (kc_upper - kc_lower)` | Volatility |
| 33 | `volume_sma` | `volume / volume.rolling(20).mean()` | Volume |
| 34 | `volume_ratio` | `volume / volume.rolling(5).mean()` | Volume |
| 35 | `obv` | On-Balance Volume (normalised by 1M) | Volume |
| 36 | `adx` | Average Directional Index (14 periods) | Trend Strength |
| 37 | `plus_di` | +DI directional indicator (14 periods) | Trend Strength |
| 38 | `minus_di` | -DI directional indicator (14 periods) | Trend Strength |
| 39 | `price_vs_sma50` | `close / sma_50 - 1` | Trend Strength |
| 40 | `price_vs_sma200` | `close / sma_200 - 1` | Trend Strength |
| 41 | `atr_pct` | `atr / close` | Trend Strength |
| 42 | `zscore_20` | `(close - sma_20) / std_20` | Mean Reversion |
| 43 | `dist_52w_high` | `(close - high_52w) / high_52w` | Mean Reversion |

---

## Category descriptions

### Price (indices 0–3)

Raw price-derived features that capture short-term dynamics without indicator lag:

- **Return / Log Return**: The most direct signal — did the price go up or down today? Log returns are additive and closer to normally distributed, making them better targets for regression.
- **Volatility (5d)**: Short-term realised volatility. High volatility means the model should be less confident in point predictions.
- **ATR**: Average True Range accounts for gaps between sessions (|high−low|, |high−prev_close|, |low−prev_close|). Better measure of actual intraday risk than simple high−low.

### Trend (indices 4–13)

Moving averages at multiple timescales capture the dominant trend direction:

- **SMA (Simple Moving Average)**: Uniform weight over the window. Slower to react than EMA.
- **EMA (Exponential Moving Average)**: Geometric decay — recent prices matter more. Faster to signal trend changes.
- Having both SMA and EMA at the same period (e.g. SMA 20 and EMA 20) lets the model learn their divergence as a signal.

### Momentum (indices 14–19)

Whether the price is accelerating or decelerating:

- **RSI**: Compares average gains vs average losses over 14 periods. Values >70 = overbought, <30 = oversold.
- **Fast RSI (7d)**: Same calculation, shorter window — more sensitive, noisier.
- **Stochastic RSI**: RSI of RSI — even more sensitive oscillator for short-term reversals.
- **Momentum**: Raw price difference. Captures the same concept as ROC but in price space rather than percentage space.
- **ROC**: Percentage change over 10 periods — normalises for price level.
- **Williams %R**: Inverse of Stochastic K — shows where the close is relative to the high-low range. -100 = at the low, 0 = at the high.

### Oscillators (indices 20–25)

Mean-reverting indicators that measure overbought/oversold conditions:

- **MACD**: Fast EMA minus slow EMA. Crossing zero = trend change. The histogram (MACD − signal) is often the most actionable signal.
- **Stochastic K/D**: %K is where today's close falls within the 14-day range. %D is a 3-day SMA of %K (smoother). Crossovers and divergences are common signals.
- **CCI**: (Price − SMA) / (0.015 × MAD). Normalised deviation from average price. ±100 are the traditional overbought/oversold levels.

### Volatility (indices 26–32)

Band-based indicators that frame price relative to expected range:

- **Bollinger Bands**: Based on a 20-period SMA ± 2 standard deviations. `%B` = 0 when price is at the lower band, 1 when at the upper. Values outside 0–1 indicate breakouts.
- **Keltner Channels**: Based on EMA20 ± 1.5 × ATR. Uses ATR for bandwidth instead of price standard deviation — more stable in volatile markets. When Bollinger Bands contract inside Keltner Channels, a breakout is often imminent (squeeze).

### Volume (indices 33–35)

Volume relative to its own history reveals accumulation/distribution:

- **Volume SMA ratio**: Volume relative to its 20-day average. Spikes often accompany trend changes.
- **Volume Ratio (5d)**: Same but shorter window — more sensitive to recent activity.
- **OBV (On-Balance Volume)**: Running total: +volume on up days, −volume on down days. Normalised by 1M to keep values in a reasonable range. OBV trending in the same direction as price confirms the trend; divergence warns of reversal.

### Trend Strength (indices 36–41)

How strong and sustained the current trend is:

- **ADX**: Measures trend strength regardless of direction. <20 = weak/choppy, >40 = strong trend.
- **+DI / −DI**: Directional indicators. When +DI crosses above −DI = bullish; below = bearish. ADX confirms whether the crossover is meaningful.
- **Price vs SMA50/200**: Distance from the 50- and 200-day averages as a fraction of price. Captures where we are in the medium/long-term trend.
- **ATR%**: ATR as a percentage of price — normalises volatility for comparison across instruments with different price levels.

### Mean Reversion (indices 42–43)

How far price has stretched from its "normal" level:

- **Z-Score (20d)**: Standard deviations from the 20-day mean. Values beyond ±2 often revert.
- **Distance from 52-week high**: Negative means price is below its annual high. Captures drawdown depth — a common feature for reversion strategies.

---

## Why no lookahead?

Every feature is computed using `df.iloc[:t+1]` — only data available at time `t`. The target (next-day return) is always `t+1`. This ensures no information leakage from the future into the model, and performance measured on held-out data reflects real deployable accuracy.

---

## Normalisation

After all 44 features are extracted for a training batch:

1. **Z-score**: Subtract training-set mean, divide by training-set std. Mean and std are stored in the checkpoint.
2. **Clamp**: `torch.clamp(x, -10, 10)`. Prevents extreme indicator values (e.g. a 1000-day ATR or OBV spike) from saturating activations and producing zero gradients.

At inference time, the same `scaler_mean` and `scaler_std` from the checkpoint are applied before feeding the model.

---

## See also

- [Architecture](ARCHITECTURE.md) — how the model processes these 44 features
- [Training guide](TRAINING.md) — data pipeline and normalisation details
