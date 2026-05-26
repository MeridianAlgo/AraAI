# Changelog

All notable changes to Meridian.AI are documented here, from the first commit to the latest release.

---

## [v5.2.2] — 2026-05-26 — Step-based LR schedule and a real training budget

**The model can finally learn again.** With v5.2.1 the pipeline was stable and pushing, but the loss curve was flat. The cause was the learning rate schedule, not the data.

### Root cause

The schedule was a two epoch linear warmup followed by cosine annealing, and `scheduler.step()` was called once per epoch. A capped run (`--max-steps 70`) stops well inside the first epoch, so the scheduler stepped at most once and the learning rate stayed pinned at the `0.1x` warmup floor (`5e-5`). The cosine phase never ran, so every run was a tiny constant rate nudge and the loss barely moved.

### Fix

When `--max-steps` is set, the schedule is now built in step units and advanced once per optimizer step:

- Warmup over the first 10 percent of steps (`0.1x` to `1x` of base LR).
- `CosineAnnealingLR` over the remaining steps, down to `eta_min = 0.05 * lr`.

For a 300 step run the LR now ramps to its `5e-4` peak by step 30 (10x higher than the old stuck value) and anneals to `2.5e-5` by the end. Offline epoch based training (no `--max-steps`) keeps the original warm restart schedule.

### Budget

The step cap is raised from 70 to **300** in both workflows, roughly two and a half hours per run on the standard runner. The hourly cron is unchanged; the existing concurrency setting (`cancel-in-progress: false`) keeps at most one run in progress and one pending, so longer runs simply shift the effective cadence instead of piling up.

---

## [v5.2.1] — 2026-05-26 — Batched validation forward pass (the real OOM fix)

**The push-finally-lands release.** v5.2.0 added the safety-save so a valid `.pt` always hit disk at the step limit — but Hugging Face *still* hadn't updated since 2026-05-15. Tracing the actual runner log showed the cause was not a time-budget SIGTERM at all:

```
09:11:01  [safety-save] OK — 70 steps written
09:11:29  ##[error]The runner has received a shutdown signal...
09:11:30  ##[error]The operation was canceled.
```

The runner was **reclaimed ~28 s after the safety-save**, during the post-training validation. Because that is a *job-level* kill, every `if: always()` step — `Verify`, **`Push to Hugging Face`**, `Pipeline Summary` — was skipped, so the safety-saved checkpoint never left the runner.

### Root cause

`AdvancedMLSystem.train` ran the post-training validation as a **single forward pass over 4096 samples** (twice: EMA + main model). Training only forwards a ~64-sample micro-batch, so validation allocated roughly 64× the activation memory through the attention/Mamba/MoE stack and OOM-killed the CPU runner the instant validation started. Deterministic — every hourly run died at the same point.

### Fix

New `AdvancedMLSystem._predict_in_batches(model, X, batch_size=256)` runs validation forwards in fixed-size chunks and concatenates the predictions, holding peak memory at training levels. Both the per-epoch validation (4096) and the final-accuracy pass (2048) now go through it. The runner survives, the script exits cleanly, and the Verify + Push steps finally run.

Side effect: the per-epoch `val_loss` / `direction_accuracy` and `summary.*` Comet metrics — which previously died with the runner — now log, and `experiment.end()` flushes cleanly.

---

## [v5.2.0] — 2026-05-23 — Single-job CI pipeline + per-step Comet curves + safety-save

**The HF-push-actually-works release.** v5.1.0 tightened the time budget but a different failure mode kept biting: the runner was SIGTERMed *after* the step-limit print but *before* the model hit disk, leaving the artifact step with nothing to upload. Hugging Face hadn't received a new checkpoint in 8 days even though every training run looked successful in the Actions UI. v5.2 traces the silent failure end-to-end and fixes the four pieces that allowed it to keep happening.

### Single-job pipeline

`meridian-forex.yml` + `meridian-stocks.yml` (four jobs each: setup → train → deploy → cleanup) collapse into one job per workflow (`forex.yml` + `stocks.yml`). The `.pt` never leaves the runner that produced it — no `actions/upload-artifact` + `actions/download-artifact` dance.

Why this matters: in the old layout, when the train step exited 143, the conditional `if: always() && hashFiles(...)` on Upload Model evaluated against an empty `models/` (because the script was killed before save). The artifact never existed, deploy's `actions/download-artifact` raised "Artifact not found", and the HF push silently skipped. In the single-job layout, the Push step runs on the same filesystem the safety-save just wrote to — `hashFiles` succeeds, the upload happens, HF gets the checkpoint even if the train step exited non-zero.

### Safety save before validation

`AdvancedMLSystem.train` now calls `_save_model()` *immediately* when `step_limit_reached` fires, before the 4096-sample CPU validation, before Comet `log_metrics`, before the final `log_model` 132 MB upload. The existing atomic write (`.tmp` + `fsync` + `os.replace`) means this can never produce a truncated checkpoint.

Result: even if the runner is killed during validation or the Comet upload, a valid `.pt` is already on disk and the next step pushes it to HF.

### Per-step Comet metrics

Per-epoch `log_metrics` never fired on 70-step CI runs (an epoch is ~187 optimizer steps, so step-limit always lands mid-first-epoch). The Comet dashboard showed parameters but zero curves. v5.2 logs per optimizer step:

- `step/train_loss` — raw per-step loss
- `step/learning_rate` — current LR from the scheduler
- `step/grad_norm` — pre-clip gradient L2 norm
- `step/elapsed_sec` — seconds since `train_start`

70 data points per run, every run, every project — real training curves on the dashboard at last.

### Per-step CI stdout

Every 5 steps (plus step 1 and the final step) the training loop now prints:

```
  step 5/70 | loss=0.0412 | lr=4.8e-04 | grad=0.831 | t=125s
```

flushed immediately. Live progress in the Actions log instead of one "Step limit reached" line at the end.

### Renames

- Workflows: `meridian-forex.yml` → `forex.yml`, `meridian-stocks.yml` → `stocks.yml`
- Scripts: `train_forex_model.py` → `train_forex.py`, `train_stock_model.py` → `train_stocks.py`, `push_elite_models.py` → `push_to_hf.py`

`git mv` was used so history is preserved.

### No step timeouts on the Train Model step

Only `--max-steps 70` governs when training stops. Job-level cap stays at 360 min (GitHub Actions maximum for public repos). Post-training validation + Comet flush no longer race against a 38- or 55- or 75-minute step deadline.

### Compatibility

State-dict layout is unchanged from v5.1.0. Existing checkpoints load and continue training — no cold restart, no week-long climb back to the current accuracy. `_MIN_LOADABLE` stays at `(4, 1)`.

### Tags

`v5.2.0`, `MeridianModel-2026`, `type:stock|forex`.

---

## [v5.1.0] — 2026-05-21 — Hardened CI + full Comet telemetry + HF legacy migration

**The reliability release.** Diagnoses and fixes the recurring "The operation was canceled" CI failure, makes every training run fully auditable in Comet, and reorganises the Hugging Face repository so v5 checkpoints live next to a `legacy/` archive of pre-v5 artifacts.

### CI cancellation root-cause fix

The forex/stocks workflows were ending with `Error: The operation was canceled.` at ~44m every run. The training script self-stops at the `--max-time` boundary, but the post-loop **validation + Comet flush + `_save_model`** block was running over the boundary and hanging on either a network call (Comet/HF) or a non-atomic `torch.save`, after which the runner was SIGTERM'd and the EMA checkpoint was lost.

Four-part fix:

1. **SIGTERM/SIGINT handler** in `AdvancedMLSystem.train` — sets a `shutdown_requested` flag the inner step-loop polls. On signal we always break out, run the final save, and exit cleanly. The handler is no-op'd on non-main-thread imports so it doesn't break callers.
2. **Atomic checkpoint write** in `_save_model` — serialises to `<path>.tmp`, `fsync`s, then `os.replace`s into place. A killed runner can no longer leave a half-written `.pt` file.
3. **Comet calls wrapped in `try/except`** — every per-epoch metric log, the one-shot parameter log, and the final summary `experiment.end()` are isolated; a hung HTTP request degrades to a warning instead of stalling the loop.
4. **Tightened time budget** — `--max-time` cut from 45 → 35 minutes; per-step `timeout-minutes: 38` so the runner's own kill arrives *after* the script has gracefully exited.

### Comprehensive Comet ML telemetry

`meridian-ai-stock-v5` and `meridian-ai-forex-v5` projects now receive, per experiment:

- **Dataset audit**: total rows, unique symbols, date range, average rows/symbol, plus a per-symbol `dataset.symbol.<SYM>` other-field with row count and date range, plus a `training_symbols.txt` asset with the symbol list.
- **Architecture summary**: model version, arch name, param count, input size, seq len, dim, layers, heads, KV heads, experts.
- **Training config**: every hyperparameter the run was started with — batch size, effective batch, gradient accumulation steps, LR, weight decay, optimizer, scheduler, loss, EMA decay, validation split, target clip, time/step budgets.
- **Target distribution**: min, max, mean, std, median, percent positive, normalisation mode.
- **Feature stats**: scaler L1 norm, mean of std, data memory footprint.
- **System info**: Python version, torch version, platform, CPU count, total RAM, process RSS.
- **Per-epoch**: train loss, val loss, EMA val loss, LR, direction accuracy, precision, recall, F1, epoch time, total elapsed, gradient norm, weight norm, process RSS, patience counter, best val loss.
- **Final summary**: best val loss, direction accuracy, F1, epochs completed, global steps, training time, whether shutdown was triggered by signal and which one.
- **Tags**: `v5.1.0`, `MeridianModel-2026`, `type:stock|forex`, `params:11M`.

### Hugging Face reorganisation

- Anything below v5 on `meridianal/ARA.AI` moves to `legacy/<filename>`.
- New helper `scripts/migrate_hf_legacy.py` does the inspection/move (idempotent, supports `--dry-run`).
- `scripts/push_elite_models.py` continues to write current models to `models/`.
- Model card refreshed to document the new layout, point at the Comet projects, and list every v5.1 hardening.

### Env handling

- `scripts/train_*.py` and `scripts/push_elite_models.py` accept both CI uppercase secrets (`HF_TOKEN`, `COMET_API_KEY`) **and** the lowercase keys in `.env` (`huggingface_token`, `comet_ai_token`).
- `load_dotenv()` called at script start so local runs work without exporting variables.

### Versions

- `pyproject.toml` and `meridianalgo/__init__.py` bumped to `5.1.0`.
- New constants `MODEL_VERSION = "5.1.0"` and `ARCHITECTURE_NAME = "MeridianModel-2026"` in `large_torch_model.py`; checkpoints saved with these values.
- Loader version check rewritten to compare version tuples (so `"5.1.0"` and `"10.0"` sort correctly).
- `revolutionary_model.py` shim now emits a `DeprecationWarning`; will be removed in v6.

---

## [v5.0.0] — 2026-05-15 — MeridianModel + Hourly CI

**The correctness release.** Seven silent training bugs fixed, architecture renamed, CI moved to hourly.

### Training bugs fixed

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | Validation ran after `break` | `best_val_loss` always `None`; direction accuracy never recorded | Moved validation block before time-limit break |
| 2 | Double normalisation | Features normalised twice (loader + training loop) | Removed redundant second normalisation pass |
| 3 | FP16 on CPU | `autocast(dtype=float16)` crashes on CPU (no tensor cores) | CPU uses `bfloat16`; `float16` only on CUDA |
| 4 | NaN patience | Early stopping triggered on NaN loss before recovery | `math.isfinite` guard; NaN epochs skip patience counter |
| 5 | Feature saturation | Raw indicators hitting `±1e6` → zero gradients | `torch.clamp(..., -10, 10)` after z-score normalisation |
| 6 | Slow Mamba scan | Python loop over sequence length (~18 min/epoch); hidden state leaked across batches | Batched matrix ops; `h` reset per sample |
| 7 | Hardcoded `use_mamba=True` | Checkpoint always saved `use_mamba=True` regardless of model config | Reads `use_mamba` from the live model layer before saving |

### Architecture rename

- **`MeridianModel`** replaces `RevolutionaryFinancialModel` as the canonical class name
- New file `meridianalgo/meridian_model.py`; `revolutionary_model.py` kept as a zero-breaking backward-compat shim
- Checkpoint version `"5.0"`, architecture string `"MeridianModel-2026"`
- Both old (`"RevolutionaryFinancialModel-2026"`) and new strings accepted on load

### Performance

- **CPU default config**: `dim=256`, `num_heads=4`, `use_mamba=False`, `mamba_state_dim=4` → ~11M params, ~27 min/epoch at 60K samples
- LR warmup: 2-epoch linear ramp (10% → 100%) before cosine annealing — stabilises early training
- Sample cap: 60K most-recent rows (down from 100K) — fits CI window with buffer

### CI

- Training schedule: **every 6 hours → every 1 hour** (stocks at `:00`, forex at `:30`)
- Concurrency groups prevent pile-up; queued runs wait rather than cancel
- `mamba_state_dim` persisted in checkpoint for faithful round-trip reload

### Tests

- `test_architecture_matches_readme` no longer hardcodes `dim=384`/`num_heads=6`/`version="4.1"` — uses range + set checks
- All test files and `conftest.py` import from `meridianalgo.meridian_model`
- `mamba_state_dim` threaded through all test fixtures (defaults to 16 for pre-v5 checkpoints)

---

## [v4.1.2] — 2026-04-07 — Pipeline fix + lint

- **Fix**: Resolved `UnboundLocalError` on variable `X` after `del` at the metadata-save step — training was succeeding but the run was reported as failed
- **Fix**: Removed unused `mem_model_mb` variable (`ruff F841`)
- **Model card**: Updated Hugging Face model card with correct v4.1 architecture specs

---

## [v4.1.1] — 2025-12-04 — Daily CI + test tagging

- **CI**: Switched training schedule to daily (one run per day per asset type)
- **CI**: 30-minute training cap per run; 60-minute job timeout
- **Tests**: Tagged test suite — `v4.1.1` cut after tests pass

---

## [v4.1.0] — 2025-11-25 — New architecture + documentation overhaul

**The architecture release.** Rewrote the model from a simple LSTM/CNN hybrid to a full hybrid transformer-SSM system.

### Architecture (v4.1)

- **Mamba-2 SSM** — selective scan for linear-time long-range sequence modeling
- **Grouped Query Attention (GQA)** — 6 heads, 2 KV heads; ~3× KV cache reduction
- **QK-Norm** — stability improvement for attention at depth
- **Rotary Position Embeddings (RoPE)** — relative temporal encoding
- **Mixture of Experts (MoE)** — 4 SwiGLU experts, top-2 routing; regime-specific specialization
- **RMSNorm + Layer Scale** — per-block normalisation for training stability
- **Stochastic Depth** — drop-path regularisation
- ~45M parameters at `dim=384`, 6 layers, 6 attention heads

### Training improvements

- **BalancedDirectionLoss** — 60% Huber regression + 40% class-weighted BCE for direction; replaces raw MSE
- **EMA weight averaging** — decay 0.999; produces smoother, better-generalizing checkpoints
- **Gradient accumulation** — effective batch 256 from micro-batches
- **Data augmentation** — Gaussian noise (0.5%) + random timestep masking (5%)
- **Multi-timeframe data** — daily + 2yr hourly + 5yr weekly per symbol (3× more samples)
- **Version gating** — stale pre-v4.1 checkpoints never loaded; train fresh if invalid
- **Comet ML** — replaces WandB for experiment tracking

### Documentation

- Added `SECURITY.md`, `CONTRIBUTING.md`, `TESTING.md`, `DOCS_INDEX.md`
- Full documentation overhaul: removed inaccuracies, added architecture diagrams, professional formatting

---

## [v3.2.2] — 2025-11-24 — Accelerate integration

- **Accelerate**: Integrated HuggingFace Accelerate library for hardware-agnostic training (CPU/CUDA/MPS)
- **CPU limiting**: Granular CPU usage cap at 80% to avoid OOM kills in CI
- **Mixed precision**: Auto-detected via Accelerate

---

## [v3.2.1] — 2025-11-23 — Repository cleanup

- Removed stale `.kiro/specs/` directory
- Cleaned up `__pycache__` and `.pytest_cache` directories from tracking

---

## [v3.0.1] — 2025-10-24 — Major reorganization

- **Renamed**: `run_ara.py` → `ara.py` (main CLI entry point)
- **Docs**: Moved all documentation to `docs/` folder (`CONTRIBUTING.md`, `CREDITS.md`, `CI_CD_IMPROVEMENTS.md`)
- **Training**: Trains only on target ticker with maximum historical data (e.g. AAPL: 30+ years / 11K samples)
- **README**: Complete rewrite with project structure diagram, emoji navigation, linked documentation

---

## [v3.0.0] — 2025-10-22 — Workflow syntax fix

- **Fix**: Replaced complex regex version-extraction with simple import statement in CI workflows — resolved `SyntaxError` in auto-release and version-bump workflows

---

## [v2.2.1] — 2025-10-08 — Project restructure

- Restructured project layout — moved source files into proper package directories
- Updated documentation to match new layout
- Improved import structure

---

## [v2.2.1-Beta] — 2025-09-21 — README improvements

- Updated README with clearer installation steps and usage examples
- Added contribution guidelines

---

## [v2.2.0-Beta] — 2025-10-18 — ULTIMATE ML System + CI/CD

**The automation release.** Full CI/CD pipeline, separate stock and forex workflows, experiment tracking.

### CI/CD pipeline

- Separate `hourly-train-stock.yml` and `hourly-train-forex.yml` workflows
- `lint.yml` for automated code quality checks (isort + black + ruff)
- Hourly training schedule for both asset types
- **Comet ML** integration for experiment tracking; **WandB removed**

### Model

- Unified stock and forex models (~4.2M parameters)
- 44+ technical indicators
- Transformer + CNN-LSTM hybrid architecture
- Smart sampling: random symbol selection for diverse learning

### Dependencies

- Added: `comet-ml`, `isort`, `black`, `ruff`
- Removed: `wandb`

---

## [v1.0.0-Beta] — 2025-08-18 — First public release

**Initial beta release of Meridian.AI (then "Ara AI").**

- Stock analysis platform using Yahoo Finance data (no API keys required)
- Direct `ara.py` entry point for new users
- Improved error handling and user experience
- PyPI package `meridianalgo` v0.3.0 released
- Windows-compatible Unicode encoding in installer scripts
- Supports CPU and GPU (CUDA/MPS/DirectML)

---

## Pre-release history

| Date | Milestone |
|------|-----------|
| 2025-08 | Initial commit: Ara — AI Stock Analysis Platform |
| 2025-08 | Pure Yahoo Finance implementation (no API keys needed) |
| 2025-08 | Enhanced accuracy system with intelligent failsafes |
| 2025-08 | Critical runtime fixes; improved prediction reliability |
| 2025-08 | `meridianalgo` v0.3.0 published to PyPI |
| 2025-08 | Ara AI v2.0.0 — complete system overhaul |
| 2025-08 | 7-day cycle prediction system with fresh-start retraining |

---

## License

MIT License. See [LICENSE](../LICENSE) for details.
