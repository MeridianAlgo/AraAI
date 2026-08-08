---
library_name: sklearn
license: mit
tags:
- finance
- trading
- time-series
- stock-prediction
- gradient-boosting
- cross-sectional
---

# Ara.AI — Stock Ranking Models

This repository hosts two generations of model. **v8 is current**; the v7
transformer checkpoints are kept for reproducibility only.

| file | model | status |
|---|---|---|
| `models/ara_v8_stocks.joblib` | Ara.AI v8, cross-sectional gradient-boosted ranker | **current** |
| `models/ara_v8_stocks.json` | v8 training metadata | current |
| `models/Meridian.AI_Stocks.pt` | v7 transformer, absolute next-day return | frozen |
| `models/Meridian.AI_Forex.pt` | v7 transformer, forex | frozen |

Source: <https://github.com/MeridianAlgo/AraAI> · Design notes:
[`docs/ARA_V8.md`](https://github.com/MeridianAlgo/AraAI/blob/main/docs/ARA_V8.md)

## What v8 predicts

For each stock on each day, the **next-day return relative to the universe
average that day** — not the absolute return. The output is a ranking; the
intended use is a dollar-neutral long/short book (long the top names, short the
bottom ones).

This matters for reading the numbers: a score of `+0.004` means "expected to
beat the universe average by about 40 bp tomorrow", not "expected to rise
0.4%". A market-neutral model does not forecast market direction and should not
be compared against an always-up baseline.

## Architecture

- `HistGradientBoostingRegressor` × 3 seeds, averaged (400 iterations, 15 leaf
  nodes, lr 0.03, L2 1.0, `max_features` 0.7, no early stopping)
- ~40 scale-invariant daily features per (symbol, date): multi-horizon returns,
  realized vol and vol-normalized shocks, SMA distances, intraday range/gap
  structure, ATR, Wilder RSI, volume z-scores, 52-week distances, calendar
- 8 of those additionally as cross-sectional within-day percentile ranks
- Target: cross-sectionally demeaned forward return, winsorized at 4 per-day σ
- No price or volume levels anywhere (they encode symbol identity, not signal)

## Performance

Expanding-window walk-forward, 4 folds over 2025-06-02 → 2026-06-08, retrained
from scratch before each fold with a 1-day embargo between train and test.

| metric | value | reference |
|---|---|---|
| mean daily rank IC | +0.0126 | 0.0 = no skill |
| IC t-statistic | +1.08 | > 2 would be significant |
| IC hit rate | 55.1% of days | 50% = no skill |
| long/short spread, top-5 vs bottom-5 | +12.6 bp/day | — |
| long/short Sharpe, annualized, **pre-cost** | +1.25 | — |
| 1-day reversal baseline | IC +0.0026, −8.6 bp/day | v8 beats it |
| direction accuracy on residual | 50.57% | 50% = no skill |

The signal is positive in all four folds and across four independent seeds, and
it beats the naive reversal baseline. **It is not statistically significant**
(t = 1.08 over 256 test days), and the pre-cost Sharpe would not survive daily
turnover on a five-name-per-side book at retail commissions. Treat this as a
small measured tilt, not a trading system.

### v7, for comparison

Trained on data before 2025-06-01 and evaluated on the year after:

| v7 model | direction accuracy | always-up baseline | return MAE | zero-pred MAE |
|---|---|---|---|---|
| Stocks | 50.23% | 51.44% | 0.0127 | 0.0127 |
| Forex (1-day embargo) | 48.68% | 52.02% | 0.0031 | 0.0030 |

v7 had no edge and its magnitude forecast sat exactly on the
zero-prediction floor. Any higher figure in older documentation came from a CI
checkpoint that had trained through its own evaluation window.

## Limitations

- **Stocks only.** 22 FX pairs is too thin a cross-section to rank, and the
  source FX daily bars leak next-day information through day-t high/low.
- **50-name universe.** Few independent bets per day; this is the main reason
  the t-statistic is small.
- **Daily close-to-close only.** No intraday, no multi-day horizon.
- **Pre-cost.** No transaction costs, slippage, borrow, or capacity modeling.
- **Survivorship.** The symbol list is today's large caps; delisted names are
  absent from the history.

## Usage

```python
import joblib

p = joblib.load("ara_v8_stocks.joblib")   # {"models": [...], "features": [...], "meta": {...}}
# Build features with `ara.make_dataset` from the repo, then:
# scores = np.mean([m.predict(X) for m in p["models"]], axis=0) / 100.0
```

The repository provides the whole path:

```bash
pip install -r requirements.txt
python -m ara predict --db-file training.db --model-path models/ara_v8_stocks.joblib
```

## Disclaimer

Research and educational use only. Not financial advice. Past performance does
not guarantee future results, and the figures above are pre-cost and not
statistically significant. Never trade with money you cannot afford to lose.
