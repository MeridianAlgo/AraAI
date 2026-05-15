# Changelog

All notable changes to Meridian.AI are documented here, from the first commit to the latest release.

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
