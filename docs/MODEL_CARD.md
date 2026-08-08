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

Expanding-window walk-forward, 4 folds over 2025-06-02 → 2026-08-06, retrained
from scratch before each fold with a 1-day embargo between train and test.
99 symbols, median 70 names per day, 297 test days.

| metric | value | reference |
|---|---|---|
| mean daily rank IC | +0.0205 | 0.0 = no skill |
| IC t-statistic | +2.26 | > 2 is the significance bar |
| IC hit rate | 56.6% of days | 50% = no skill |
| long/short spread, top-5 vs bottom-5 | +18.6 bp/day | — |
| long/short Sharpe, annualized, **pre-cost** | +1.58 | — |
| 1-day reversal baseline | IC +0.0025, −8.4 bp/day | v8 beats it |
| direction accuracy on residual | 51.03% | 50% = no skill |

Positive in all four folds and stable across seed groups (IC +0.0204 to
+0.0209, t 2.25–2.30). **t = 2.26 clears the conventional bar only just**, on a
single window, and one fold (+0.0405) carries much of the average while the
other three sit near +0.013 with t < 1 individually. The long/short figures are
pre-cost and would not survive daily turnover on a five-name-per-side book at
retail commissions. Treat this as a small measured edge, not a trading system.

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
- **99-name universe.** Median 70 tradeable names per day and 5 per side means
  few independent bets, which is why the long/short P&L is much noisier than
  the rank IC.
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
