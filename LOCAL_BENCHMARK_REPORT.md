# Local Benchmark & Training Audit (2026-06-12)

Local-only evaluation of the Meridian model. Nothing in this report has been
pushed to GitHub or HuggingFace. All numbers are **out-of-sample**: models
were trained on `training_pre2025H2.db` (data strictly before 2025-06-01,
built with `scripts/make_timesplit_db.py`) and evaluated on the year after
(`scripts/benchmark_model.py --holdout-start 2025-06-01`).

## Why the old "benchmarks" were meaningless

The CI-trained HF checkpoint scores 79.86% direction accuracy when evaluated
on recent data — but it *trained through* that data (hourly CI retraining
with a most-recent-60K-samples cap), so that number is in-sample memory, not
skill. Every number below is from a model that never saw the holdout year.

## Honest results (holdout: 2025-06-02 → 2026-06-08/09)

| | v6 forex | v7 forex (embargoed) | v6 stocks | v7 stocks |
|---|---|---|---|---|
| direction accuracy | 52.08%* | 48.68% | 50.18% | 50.23% |
| always-up baseline | 52.02% | 52.02% | 51.44% | 51.44% |
| edge vs best baseline | +0.05* | −3.34 | −1.26 | −1.21 |
| return MAE | 0.0112 | 0.0031 | 0.0150 | 0.0127 |
| zero-prediction MAE (floor) | 0.0030 | 0.0030 | 0.0127 | 0.0127 |

v6 = pristine pre-v7 code (git 201840c, trained/evaluated from the
`AraAI_v6eval` worktree). v7 = current tree. Same data, same 2000-step CPU
budget (~7 min on a 16-core desktop). Stock direction accuracy is stable
across three independent v7 runs (50.23–50.30%).

**\* All un-embargoed forex numbers are contaminated — see below.** v7
forex trains and evaluates with a 1-day embargo (`embargo_days=1`), which
blocks the artifact; what remains is the honest result: **no daily FX
direction edge exists in this data**, and the model's small bearish tilt
makes it slightly worse than always-up. Un-embargoed v7 forex runs scored a
seed-dependent 56.9% / 76.7% / 78.5% — pure artifact readout, not skill.

What v7 genuinely fixed (visible in the trustworthy stock/magnitude rows):

- **Magnitude calibration.** v6 predicted absurd magnitudes (live HF forex
  checkpoint: ~17%/day predicted moves; honest v6 forex MAE 3.7× worse than
  predicting zero). v7's MAE sits at the zero-prediction floor for both
  asset classes — the regression head is now calibrated.
- **The 0.277 dead-prediction equilibrium is gone.** v6's loss bottomed at
  0.6·(tiny Huber) + 0.4·log(2) ≈ 0.277 because raw returns (~0.005) were
  ~200× too small for SmoothL1's quadratic region and the BCE logits
  saturated. The loss now works in percent units (`return_scale=100`).
- **Direction accuracy on stocks is ≈ the always-up baseline, not above
  it.** On clean daily OHLCV bars with per-symbol technical indicators, a
  next-day direction edge over the drift baseline is what market-efficiency
  literature predicts: ~zero. Claims above that level from this data should
  be treated as leakage until proven otherwise (see below).

## CRITICAL: the forex bars in training.db are internally inconsistent

A linear regression using **only day-t** `high/close`, `low/close`,
`open/close` predicts the day-t→t+1 close-to-close return with
**corr +0.81 / 81% sign accuracy** on the holdout year
(`scripts/diag_feat_corr.py`). On stock bars the same probe gives a null
result (corr +0.06 / 51.9%).

Per-pair diagnostics (`scripts/diag_bars.py`): today's range midpoint vs
close correlates ~+0.40 with tomorrow's return; ~38% of forex bars have
open == close to within 1e-4; 67% of next-day closes fall inside today's
range. The daily `*=X` candles are built from indicative quotes whose
high/low span a different (later) window than the stored close — so each
bar's high/low leaks information about the *next* close.

Consequences:

- **Any forex backtest on this DB is invalid.** A v7 dev run that selected
  checkpoints by validation direction accuracy reached a fake **76.7%**
  holdout score (corr(pred, y) = 0.73, 92.6% on big moves) purely by
  reading the artifact — verified by a skip-day test (accuracy on the t+2
  return collapses to 49%). The v7 forex 56.93% above is partially
  artifact-assisted; the honest forex edge is unknown until the data source
  is fixed.
- **The CI forex pipeline was training on contaminated data.** Models it
  pushed would look good on any same-source evaluation and fail live.
- **Mitigation implemented:** forex now trains and benchmarks with a 1-day
  embargo (`embargo_days=1` in `scripts/train_forex.py`, default
  `--embargo 1` for forex in `scripts/benchmark_model.py`) — the input
  window ends at t−1, so the contaminated day-t bar can never be read.
  The full fix remains re-ingesting forex history from a source with
  consistent bar semantics (and ideally intraday-validated closes).

## Training pipeline fixes shipped in this branch (local, not pushed)

1. `BalancedDirectionLoss` rescaled to percent units; class-weighted BCE
   made opt-in (`balance_classes=False` default) — weighting recentred the
   direction optimum at 50/50, erasing the learnable up-drift prior and
   collapsing one run to a constant-bearish 48.7%.
2. Checkpoint dropout bug: `_save_model` hardcoded `dropout: 0.15`, so
   every warm-started CI run silently retrained with dropout the config had
   deliberately set to 0.0. Now saved from the live model.
3. v6→v7 feature overhaul: all 44 features scale-invariant (raw price/volume
   levels encoded symbol identity across 300× price ranges after
   cross-symbol z-scoring). `MODEL_VERSION = 7.0.0`, `_MIN_LOADABLE = (7, 0)`
   refuses v6 checkpoints (shapes identical, semantics changed).
4. Checkpoint selection stays min-EMA-val-loss (dir-acc selection tested:
   OOS-neutral on clean data, maximally exploits leakage on dirty data).
5. Unicode prints (`✓`, `⚠️`, `→`) replaced with ASCII — they crashed
   training/benchmarks on Windows cp1252 consoles.
6. CI: `scripts/hf_download.py` retries HF 429s with Retry-After + jitter
   (warm start is best-effort, always exits 0); hourly crons kept, staggered
   (stocks :00, forex :30) so the pipelines never hit HF simultaneously.
7. 1-day forex embargo in training and benchmarking (see above);
   `embargo_days` is persisted in checkpoint metadata.

## Reproduce

```
python scripts/make_timesplit_db.py --dst training_pre2025H2.db --cutoff 2025-06-01
python scripts/train_stocks.py --db-file training_pre2025H2.db --output bench_models/v7_stocks.pt --use-all-data --epochs 999 --max-steps 2000
python scripts/benchmark_model.py --model-path bench_models/v7_stocks.pt --model-type stock --db-file training.db --holdout-start 2025-06-01
python scripts/diag_feat_corr.py training.db forex 2025-06-01   # the leak probe
```
