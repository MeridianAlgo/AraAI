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

Expanding-window walk-forward, 4 folds over 2025-06-02 → 2026-08-06, retrained
from scratch before each fold with a 1-day embargo between train and test. 99
symbols, median 70 names per day, 297 test days.

| metric | v8 | reference |
|---|---|---|
| mean daily rank IC | **+0.0205** | 0.0 = no skill |
| IC t-statistic | **+2.26** | > 2 is the significance bar |
| IC hit rate | 56.6% of days | 50% = no skill |
| long/short spread (top-5 vs bottom-5) | **+18.6 bp/day** | — |
| long/short Sharpe (annualized, pre-cost) | **+1.58** | — |
| 1-day reversal baseline IC | +0.0025 | v8 beats it |
| 1-day reversal baseline L/S | −8.4 bp/day | v8 beats it |
| direction accuracy on residual | 51.03% | 50% = no skill |
| direction accuracy on raw return | 50.12% | always-up 51.64% |

Per-fold IC: +0.0147 / +0.0405 / +0.0114 / +0.0155 — positive in all four.
Repeating the whole backtest with different seed groups gives IC +0.0205 /
+0.0205 / +0.0209 / +0.0204 (t 2.25–2.30), so the number is not a seed draw.

**Read this carefully.** t = 2.26 clears the conventional bar, but only just,
and it is one test on one window. Fold 1 (+0.0405) carries much of the average
while the other three sit at +0.011 to +0.015 with t < 1 individually. The
long/short spread ranges +12.6 to +18.6 bp/day across seed groups — ten names a
day carries far more idiosyncratic variance than the IC it comes from, so trust
the IC and treat the P&L as an illustration, pre-cost and pre-slippage. This is
a small measured edge, not a trading system.

### What the earlier 50-symbol numbers were

Before the universe fix below, the same code measured IC +0.0126 (t 1.08) and
then +0.0089 (t 0.81) on two CI runs that differed only in which symbols got
fetched. Those numbers are superseded, not a second opinion — they were
measured on randomly drawn half-universes.

### Why the universe fix mattered so much

`scripts/fetch_and_store_data.py` used to call `random.shuffle(symbols)` and
then take the first `--limit 50` of 100. Every CI run therefore trained and
evaluated on a *different random half* of the universe, and `training.db` is
not cached between runs. Two runs of byte-identical code measured IC +0.0153
and +0.0089 for that reason alone.

Removing the shuffle and fetching all 99 symbols (PXD dropped — delisted in
2024, fetches empty) roughly doubled the cross-section and moved IC from
+0.0089/+0.0153 to a stable +0.0205.

Two distinct things caused that, and they are worth separating:

1. **More training data.** Twice the symbols, ~1.02M usable samples instead of
   ~0.47M.
2. **A more precise measurement.** Each day's IC is now a rank correlation over
   ~70 names instead of ~35. Halving the noise in each daily observation raises
   the t-statistic even when the underlying signal is unchanged — a better
   instrument, not a stronger effect.

Both are real gains, but only the first means the model got better. The honest
summary is that the 50-symbol setup was too noisy to measure this signal at
all, in either direction.

A guard is now in CI: `--min-symbols 90` fails the run if the fetched universe
shrinks, and the backtest JSON records `n_symbols` and
`median_universe_per_day` so two reports can be compared meaningfully.

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

### Universe size is still the top lever

99 symbols, median 70 tradeable names per day, is enough to measure the signal
but not much more. At 5 names per side there are still few independent bets,
which is why the long/short P&L swings so much more than the IC. Extending
the list in `scripts/fetch_and_store_data.py` to a few hundred liquid names is
the next obvious step; the fetch cost is roughly linear (99 symbols take
~2 minutes) and the walk-forward already scales fine (~105 s for 4 folds on
1.02M samples).

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
