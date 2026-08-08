# Ara.AI

### Cross-sectional daily stock ranking

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/version-2.0.0-green.svg)
![Model](https://img.shields.io/badge/model-v8-blue.svg)
[![Ara.AI v8](https://github.com/MeridianAlgo/AraAI/actions/workflows/ara-v8.yml/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions/workflows/ara-v8.yml)
[![Lint](https://github.com/MeridianAlgo/AraAI/actions/workflows/lint.yml/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions/workflows/lint.yml)

## Overview

Ara.AI v8 ranks a universe of stocks by their expected **next-day return
relative to each other**, and holds that view as a dollar-neutral long/short
book. It is a gradient-boosted tree ensemble over ~40 scale-free daily
features. It trains in about eleven seconds on a CPU, needs no GPU, and the
whole GitHub Actions pipeline — fetch, four-fold walk-forward backtest, final
fit, publish — finishes in a few minutes.

It replaces v7, a 433K-parameter transformer that predicted each stock's
absolute next-day return, retrained hourly, and had **no measured edge**:
50.23% direction accuracy against a 51.44% always-up baseline. The full
rationale, the diagnosis, and every number are in
**[docs/ARA_V8.md](docs/ARA_V8.md)**. The v7 stack is frozen under
[`legacy/`](legacy/README.md) and still runs on demand.

Trained models: [meridianal/ARA.AI](https://huggingface.co/meridianal/ARA.AI).

## What changed, in one paragraph

Most of a stock's next-day return is the *market's* next-day return, which
daily OHLCV cannot predict — so a model trained on absolute returns spends all
its capacity on noise, and "always up" beats it because the market drifts up.
v8 subtracts the universe's mean return out of the label and predicts only the
residual: which names beat their peers. That target is forecastable, the
baseline it must beat is a true 50%, and the natural output is a ranking rather
than a price. Switching from a transformer to gradient-boosted trees followed
from the same honesty: ~40 weak tabular features is the regime where trees win,
and they cost seconds instead of minutes.

## Performance

Expanding-window walk-forward, 4 folds over 2025-06-02 → 2026-06-08, retrained
before each fold with a 1-day embargo. Out-of-sample, reproducible with the
command in [Quick Start](#quick-start).

| metric | v8 | reference |
|---|---|---|
| mean daily rank IC | **+0.0126** | 0.0 = no skill |
| IC t-statistic | +1.08 | > 2 would be significant |
| IC hit rate | 55.1% of days | 50% = no skill |
| long/short spread (top-5 vs bottom-5) | **+12.6 bp/day** | — |
| long/short Sharpe (annualized, pre-cost) | +1.25 | — |
| 1-day reversal baseline | IC +0.0026, −8.6 bp/day | v8 beats it |

**Read this honestly.** The signal is positive in every fold and every seed,
and it beats the naive reversal baseline — but it is not statistically
significant (t = 1.08 over 256 test days), and a pre-cost Sharpe of 1.25 on a
five-name-per-side book would not survive daily turnover at retail commissions.
Re-run by CI on a longer window (297 days, through 2026-08-07) the IC improved
to +0.0153 (t 1.39, 56.9% hit rate) while the long/short spread fell to
+5.9 bp/day — the ranking held, the P&L on ten names a day is much noisier than
the IC. Both windows are in [docs/ARA_V8.md](docs/ARA_V8.md). The defensible
claim is "a small, consistently positive cross-sectional signal, measured with
a harness that market drift cannot game." Not a trading system.

The binding constraint is universe size: the database carries 50 mega-caps, so
each day offers ~50 ranks and very few independent bets. Widening the symbol
list in `scripts/fetch_and_store_data.py` is the highest-leverage change
available.

## Quick Start

**Requirements:** Python 3.9+. No GPU, no torch.

```bash
git clone https://github.com/MeridianAlgo/AraAI.git
cd AraAI
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# Fetch daily bars into training.db
python scripts/fetch_and_store_data.py --db-file training.db --asset-type stock --limit 50

# The honest number: walk-forward backtest (~50 s)
python -m ara eval --db-file training.db --holdout-start 2025-06-01 --folds 4

# Fit on all data and save
python -m ara train --db-file training.db --output models/ara_v8_stocks.joblib

# Rank the most recent day
python -m ara predict --model-path models/ara_v8_stocks.joblib
```

### From Python

```python
from ara import load, load_panel, make_dataset, predict

model = load("models/ara_v8_stocks.joblib")
data = make_dataset(load_panel("training.db", "stock"))
today = data[data["date"] == data["date"].max()].copy()

today["score"] = predict(model, today)          # expected return vs the universe
print(today.sort_values("score", ascending=False)[["symbol", "score"]].head(10))
```

`score` is a predicted **residual** return: +0.004 means "expected to beat the
universe average by ~40 bp tomorrow", not "expected to rise 0.4%".

## How it works

```
daily OHLCV panel  ->  build_features()  ->  one row per (symbol, date):
                                             ~40 scale-free features + xs_* day-ranks
                                                        |
                             target = fwd_ret - universe_mean(fwd_ret), winsorized at 4σ
                                                        |
                                       3x HistGradientBoosting (seed-averaged)
                                                        |
                                     predicted residual return -> daily ranking
```

Every feature is a return, ratio, z-score, or within-day rank — no price or
volume **levels**, which encode symbol identity rather than signal. Eight
features additionally carry a cross-sectional percentile rank so the model can
ask whether a name is stretched *relative to its peers today*.

The v7 pipeline fed a 30×44 tensor per sample; v8 feeds one row with lagged
returns as columns, since a tree reads `ret_21` directly and never needed the
window. That is most of the 40× memory reduction and the speedup.

## Efficiency vs v7

| | v7 | v8 |
|---|---|---|
| training time | ~7 min (2000 steps, CPU) | **~11 s** |
| CI dependencies | torch, accelerate, comet-ml (~200 MB) | numpy, pandas, scikit-learn (~50 MB) |
| CI runs per day | 48 (hourly stocks + forex) | **1** |
| model | 433,059 parameters | 3 × 400 trees, 15 leaves |

Daily bars change once a day, so 23 of every 24 hourly runs retrained on
identical data. The daily schedule is not a compromise; it is the correct one.

## The CI gate

`ara-v8.yml` runs the walk-forward backtest **before** fitting the shipped
model and fails the run if mean IC is negative (`--min-ic 0.0`). Nothing
reaches Hugging Face unless it measured a positive out-of-sample signal — which
is precisely how v7 managed to publish an edgeless checkpoint every hour for
months.

## Project Structure

```
ara/
  features.py              Vectorized panel feature engineering
  model.py                 Dataset, training, evaluation, walk-forward, persistence
  __main__.py              CLI: python -m ara train|eval|predict
scripts/
  fetch_and_store_data.py  Market data ingestion into SQLite
  push_to_hf.py            Uploads models to Hugging Face
  hf_download.py           429-aware model download
tests/
  test_ara_v8.py           Lookahead, symbol-isolation, target, roundtrip, planted-signal
.github/workflows/
  ara-v8.yml               Daily train + backtest gate + publish
  lint.yml                 Formatting and lint
  stocks.yml / forex.yml   Legacy v7 pipelines, dispatch-only
legacy/                    Frozen v7 transformer stack — see legacy/README.md
```

## Documentation

- **[Ara.AI v8](docs/ARA_V8.md)** — design rationale, measured numbers, limits
- [Legacy v7](legacy/README.md) — what it was, why it was retired, how to run it
- [Local Benchmark Report](LOCAL_BENCHMARK_REPORT.md) — the v6/v7 audit that motivated v8
- [Quick Start](docs/QUICK_START.md), [FAQ](docs/FAQ.md), [Model Card](docs/MODEL_CARD.md)
- [Changelog](docs/CHANGELOG.md)

## Forex

v8 is stocks-only. A 22-pair cross-section is too thin to rank, and the source
FX bars leak next-day information through day-t high/low — a plain regression
on day-t OHL ratios "achieves" 81% sign accuracy on that data, none of it real.
The v7 forex pipeline is frozen and dispatch-only rather than ported. Details
in [LOCAL_BENCHMARK_REPORT.md](LOCAL_BENCHMARK_REPORT.md).

## Disclaimer

This software is for research and educational purposes only. It is not
financial advice.

Trading financial instruments carries significant risk. Every prediction is a
probabilistic forecast based on historical data, and past performance does not
guarantee future results. Markets can behave in ways no model expects during
sudden shocks, liquidity crises, or structural shifts. The performance figures
above are pre-cost, pre-slippage, and not statistically significant.

You should never trade with money you cannot afford to lose. Any trading
decision you make is yours alone. MeridianAlgo and its contributors are not
liable for any financial loss that results from using this software.

The software is provided as is, without warranty of any kind. By using it you
agree to hold MeridianAlgo and all contributors harmless from any claim that
arises from your use of it. You are responsible for following all financial
regulations that apply to you.

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.

Made with care by [MeridianAlgo](https://github.com/MeridianAlgo)
