# Ara.AI v8 — cross-sectional stock ranking

v8 replaces the v7 transformer as the production stock model. This document is
the design rationale and the measured numbers. Everything here is out-of-sample
and reproducible with the commands at the bottom.

## Why v7 was retired

v7 was a 433K-parameter transformer (grouped-query attention, mixture of
experts, RoPE) reading a 30-day × 44-feature window per symbol and predicting
that symbol's next-day return. Retrained hourly in CI, published to Hugging Face.

Its honest out-of-sample result, measured on 2025-06-02 → 2026-06-08 after
training only on data before 2025-06-01:

| | v7 stocks |
|---|---|
| direction accuracy | 50.23% |
| always-up baseline | 51.44% |
| edge vs best baseline | **−1.21 pts** |
| return MAE | 0.01274 |
| zero-prediction MAE (the floor) | 0.01273 |

It was worse than a constant "up" prediction, and its return forecast was
indistinguishable from predicting zero. The architecture was not the problem.
Two other things were.

### 1. The target was mostly unpredictable

A stock's next-day return decomposes into the market's move plus the stock's
own deviation. On daily bars the market component is the large one and it is
essentially unforecastable from OHLCV. Predicting the total means spending all
model capacity on the noisy part — and it makes "always up" a strong baseline,
because the market drifts up. That is the whole reason v7 could not beat it.

**v8 predicts the cross-sectional residual**: next-day return minus the mean
next-day return across the universe that day. The market factor is subtracted
out of the label, so what is left is the part that daily bars can plausibly
speak to: which names outperform their peers. The output is a ranking, and the
natural way to hold it is dollar-neutral long/short.

### 2. Deep nets are the wrong tool for ~40 noisy tabular features

Signal-to-noise here is low and the inputs are heterogeneous scalar
indicators — the regime where gradient-boosted trees have consistently beaten
neural networks. v8 is a `HistGradientBoostingRegressor` ensemble (3 seeds
averaged). No torch, no accelerate, no GPU, no checkpoint format, no warm-start
plumbing.

## Architecture

```
daily OHLCV panel  ->  build_features()  ->  per-symbol row of ~40 scale-free features
                                             + xs_* cross-sectional rank features
                                                        |
                             target = fwd_ret - universe_mean(fwd_ret), winsorized at 4σ
                                                        |
                                       3x HistGradientBoosting (averaged)
                                                        |
                                     predicted residual return -> daily ranking
```

Features are all returns, ratios, z-scores, or within-day ranks. No price or
volume **levels**: those encode symbol identity, which is the defect that
pinned v6 at coin-flip. Eight features additionally get a `xs_` cross-sectional
rank (centered percentile within that day's universe) so the model can ask "is
this name stretched relative to its peers today", which is the question the
residual target actually poses.

Rows replaced windows: v7 fed a 30×44 tensor per sample, v8 feeds one row with
lagged returns as columns. A tree reads `ret_21` directly; it never needed the
raw window. That is the ~40× memory reduction and most of the speedup.

## Measured performance

Expanding-window walk-forward, 4 folds over 2025-06-02 → 2026-06-08, retrained
from scratch before each fold with a 1-day embargo between train and test:

| metric | v8 | reference |
|---|---|---|
| mean daily rank IC | **+0.0126** | 0.0 = no skill |
| IC t-statistic | **+1.08** | > 2 would be significant |
| IC hit rate | 55.1% of days | 50% = no skill |
| long/short spread (top-5 vs bottom-5) | **+12.6 bp/day** | — |
| long/short Sharpe (annualized, pre-cost) | **+1.25** | — |
| 1-day reversal baseline IC | +0.0026 | v8 beats it |
| 1-day reversal baseline L/S | −8.6 bp/day | v8 beats it |
| direction accuracy on residual | 50.57% | 50% = no skill |
| direction accuracy on raw return | 50.05% | always-up 51.44% |

Per-seed runs (single booster, no averaging) landed at IC +0.0099 / +0.0147 /
+0.0157 / +0.0163 — consistently positive, which is why the shipped model
averages three seeds rather than gambling on one.

### The same run on a longer window

CI re-ran this on data through 2026-08-07 (297 test days instead of 256):

| metric | 256-day window | 297-day window |
|---|---|---|
| mean daily rank IC | +0.0126 (t 1.08) | **+0.0153 (t 1.39)** |
| IC hit rate | 55.1% | **56.9%** |
| long/short spread | +12.6 bp/day | **+5.9 bp/day** |
| long/short Sharpe | +1.25 | **+0.60** |
| 1-day reversal baseline IC | +0.0026 | −0.0016 |

The ranking signal held up — IC and hit rate both improved. The *P&L* did not:
the added fold (2026-04-22 → 2026-08-06) scored IC +0.0018 and −11.0 bp/day,
roughly halving the long/short spread. Both tables are reported because
quoting only the first would be cherry-picking the window.

The gap is informative. A top-5/bottom-5 book converts a rank signal into P&L
through ten names a day, so it carries far more idiosyncratic variance than the
IC does. IC is the metric to trust here; the long/short numbers are an
illustration of what the ranking is worth before costs, not a track record.

**Read this honestly.** The IC is positive in all four folds and across every
seed, and it beats the naive reversal baseline on both IC and P&L. It is *not*
statistically significant: t = 1.08 over 256 test days. The pre-cost Sharpe of
1.25 would not survive daily turnover on a 5-name-per-side book at retail
commissions. v8's defensible claim is "a small, consistently positive
cross-sectional signal, measured with a harness that cannot be gamed by market
drift" — not "a profitable trading system".

Note that v8's raw-return direction accuracy (50.05%) is also below always-up,
and that is expected and fine: a market-neutral model is not trying to predict
market direction. Comparing it to always-up is the category error v7's metrics
were built on.

### The binding constraint is universe size

The training DB carries 50 mega-cap symbols. A 50-name cross-section gives
~50 ranks per day and, at 5 names per side, very few independent bets — which
is most of why the t-statistic is small. Widening the fetch universe (the
symbol list in `scripts/fetch_and_store_data.py`) to a few hundred liquid names
is the single highest-leverage change available, and it costs one list edit
plus a longer fetch. It has not been done here because it cannot be validated
without re-ingesting data.

## Efficiency

| | v7 | v8 |
|---|---|---|
| training time | ~7 min (2000 steps, CPU) | **~11 s** per fit (3 seeds, full panel) |
| full CI pipeline | fetch + install torch + train + gate + push | fetch + train + 4-fold backtest + push |
| CI dependencies | torch, accelerate, comet-ml (~200 MB) | numpy, pandas, scikit-learn (~50 MB) |
| CI runs per day | 48 (stocks + forex, hourly) | **1** (daily bars change once a day) |
| checkpoint size | 1.7 MB `.pt` | 0.9 MB `.joblib` |
| model params | 433,059 | 3 × 400 trees, 15 leaves |

The hourly schedule was the most expensive thing in the old pipeline and it
bought nothing: daily bars update once per day, so 23 of every 24 runs retrained
on identical data.

## The CI gate

`ara-v8.yml` runs the walk-forward backtest **before** training the shipped
model and fails the run if mean IC is negative (`--min-ic 0.0`). A degenerate
model cannot reach Hugging Face by accident, which is how v7 published an
edgeless checkpoint hourly for months. Raise `--min-ic` as the universe grows.

## Reproducing

```bash
pip install -r requirements.txt

# walk-forward backtest (the honest number) — ~50 s
python -m ara eval --db-file training.db --holdout-start 2025-06-01 --folds 4

# fit on everything and save
python -m ara train --db-file training.db --output models/ara_v8_stocks.joblib

# rank the most recent day in the DB
python -m ara predict --model-path models/ara_v8_stocks.joblib
```

The v7 stack is frozen under `legacy/` and still runs; see `legacy/README.md`.

## What v8 does not do

- **Forex.** 22 pairs is too thin a cross-section to rank, and the source FX
  bars leak next-day information through day-t high/low
  (`LOCAL_BENCHMARK_REPORT.md`). The v7 forex pipeline stays frozen and
  dispatch-only rather than being ported.
- **Intraday.** Everything here is daily close-to-close.
- **Transaction costs.** The long/short numbers are pre-cost and pre-slippage.
