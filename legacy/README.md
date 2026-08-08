# Legacy — Meridian.AI v7 (frozen)

Everything in this directory is the pre-v8 stack: the `meridianalgo`
transformer package, its training/benchmarking scripts, its tests, and its
dependency set. It is kept because the published `Meridian.AI_Stocks.pt` /
`Meridian.AI_Forex.pt` checkpoints on Hugging Face were produced by this code
and should stay reproducible.

**It is frozen.** No scheduled runs, no new features, no bug fixes beyond
keeping it importable. The production stock model is `ara/` (v8) — see
[`docs/ARA_V8.md`](../docs/ARA_V8.md) for why it was replaced and what it
measured.

## Layout

```
legacy/
  meridianalgo/     v7 transformer (GQA + MoE + RoPE), feature pipeline, utils
  scripts/          train_stocks, train_forex, benchmark_model, sanity_check_model, diag_*
  tests/            checkpoint health, inference, denormalization, directional signal
  requirements.txt  torch + accelerate + comet-ml
```

## Running it

Everything resolves relative to `legacy/`, so the scripts work unchanged:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r legacy/requirements.txt

python legacy/scripts/train_stocks.py --db-file training.db \
    --output models/Meridian.AI_Stocks.pt --max-steps 2000

python legacy/scripts/benchmark_model.py --model-path models/Meridian.AI_Stocks.pt \
    --model-type stock --db-file training.db --holdout-start 2025-06-01

pytest legacy/tests/
```

The `Stock Training (legacy v7)` and `Forex Training (legacy v7)` workflows
still exist and still work, but they are `workflow_dispatch` only.

## Why it was retired

Trained honestly (on data before 2025-06-01, evaluated on the year after) the
v7 stock checkpoint scored **50.23%** next-day direction accuracy against a
**51.44%** always-up baseline, with return MAE exactly at the
zero-prediction floor. Forex scored 48.68% against a 52.02% baseline. The
in-sample 79.86% figure that appeared in older docs came from a CI checkpoint
that had trained through its own evaluation window.

The diagnosis and the fix are in [`docs/ARA_V8.md`](../docs/ARA_V8.md); the
original audit is in [`LOCAL_BENCHMARK_REPORT.md`](../LOCAL_BENCHMARK_REPORT.md).
