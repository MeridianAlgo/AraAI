# Meridian.AI

### Real Time Financial Prediction Engine

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/version-6.0.1-green.svg)
[![Forex Training](https://github.com/MeridianAlgo/AraAI/actions/workflows/forex.yml/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions/workflows/forex.yml)
[![Stock Training](https://github.com/MeridianAlgo/AraAI/actions/workflows/stocks.yml/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions/workflows/stocks.yml)
[![Lint](https://github.com/MeridianAlgo/AraAI/actions/workflows/lint.yml/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions/workflows/lint.yml)

## Overview

Meridian.AI is a deep learning system that forecasts price movements for any stock or forex pair. It reads recent market history, turns that history into a set of technical features, and produces two things at once: a price forecast and a direction signal that estimates whether the next move is likely to be up or down.

The system retrains itself every hour through GitHub Actions and publishes each fresh checkpoint to Hugging Face, so the version you download is always the most recent one. You do not need a GPU to use it, and you do not need to run any training yourself.

The trained models live here: [meridianal/ARA.AI](https://huggingface.co/meridianal/ARA.AI).

## Architecture

The model is called MeridianModel. It is a compact transformer style network adapted for financial time series, and it carries roughly 430 thousand parameters in the configuration that ships in the hourly pipeline. Earlier versions used an 11 million parameter configuration, but a fully clean training pipeline revealed that capacity caused the model to collapse onto a trivial constant prediction; a deliberately smaller network has to extract real signal from the technical indicators in order to minimize the loss. The design favors techniques that stay stable on a CPU and train quickly, because every hourly run happens on a standard GitHub Actions runner with no GPU attached.

Each component has a specific job:

| Component | What it is | Why it is there |
|-----------|------------|-----------------|
| Attention | Grouped Query Attention with query and key normalization | Lets every timestep look at the others while sharing key and value projections, which keeps memory use low |
| Position encoding | Rotary Position Embeddings | Gives the model a sense of relative time without a fixed lookup table |
| Expert routing | Mixture of Experts with SwiGLU experts and top 2 routing | Sends each input to the two experts best suited to it, so different market conditions get different treatment |
| Activations | SwiGLU gated units | Smoother gradient flow than plain ReLU or GELU |
| Normalization | RMSNorm with layer scale | Keeps training numerically stable |
| Regularization | Stochastic depth and dropout | Reduces overfitting |
| Optional state space block | Mamba SSM with a vectorized scan | Captures long range patterns when it is switched on |
| Loss | BalancedDirectionLoss, a blend of Huber regression and binary cross entropy | Trains for price accuracy and direction accuracy together |

### Default configuration

| Setting | Value |
|---------|-------|
| Parameters | about 430 thousand |
| Hidden dimension | 96 |
| Layers | 3 |
| Attention heads | 4, with 2 key and value heads |
| Experts | 2, with top 2 routing |
| Prediction heads | 2 |
| Input features | 44 technical indicators |
| Sequence length | 30 timesteps |
| Mamba SSM | off by default on CPU |

## How a Prediction Is Made

The model reads the last 30 timesteps of market data. For each timestep it computes 44 technical indicators from the raw open, high, low, close, and volume values. After normalization, every feature is clamped to a fixed range (`[-10, 10]` in the code) so that extreme values cannot push the gradients toward zero. The cleaned features then flow through the network and out the other side as a single combined prediction.

```
Market data for any ticker or pair
        |
        v
44 technical indicators
(RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, and more)
        |
        v
Grouped Query Attention with rotary positions
        |
        v
Mixture of Experts layer (SwiGLU experts, top 2 routing)
        |
        v
Prediction heads, combined into one output
        |
        v
Price forecast and direction signal
```

The prediction heads each make their own forecast, and the model blends them with learned weights into a final number. The direction signal comes from the same forward pass, which is why the loss function trains both targets at the same time.

## Performance and Honest Expectations

This is a next-day directional model. Measured live on held-out daily data (eight large-cap stocks, 240 predictions), it scores about 57 percent directional accuracy against a roughly 55 percent always-up baseline. The mean prediction is small and positive, in line with the real daily drift of equities, and it predicts both up and down rather than collapsing onto one direction.

That is a small but genuine edge for a single day ahead. It is not a multi-day or week-ahead forecaster. Daily price direction is close to efficient, recursive multi-step forecasts compound their error quickly, and any tool that claims reliable week-ahead price prediction from price and indicator data alone is overfitting. Treat the output as a next-day directional tilt, not a crystal ball.

Earlier versions reported much higher accuracy during training but performed near chance live, because a contaminated data pipeline let the model collapse onto a near-constant downward prediction. Version 6.0 fixed the pipeline (see How Training Works) and added a sanity gate that blocks a degenerate model from ever being published.

## Quick Start

Clone the repository and install the dependencies.

```bash
git clone https://github.com/MeridianAlgo/AraAI.git
cd AraAI

python -m venv venv
source venv/bin/activate  # on Windows use: venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Predict a stock

```python
from meridianalgo.unified_ml import UnifiedStockML
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="meridianal/ARA.AI",
    filename="models/Meridian.AI_Stocks.pt",
)

ml = UnifiedStockML(model_path=model_path)
result = ml.predict_ultimate("AAPL", days=5)
print(result)
```

### Predict a forex pair

```python
from meridianalgo.forex_ml import ForexML
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="meridianal/ARA.AI",
    filename="models/Meridian.AI_Forex.pt",
)

ml = ForexML(model_path=model_path)
result = ml.predict_forex("EURUSD=X", days=5)
print(result)
```

The first call downloads the checkpoint from Hugging Face and caches it locally, so later runs start instantly.

## How Training Works

Training runs on its own, without anyone starting it. GitHub Actions launches the stock pipeline at the top of every hour and the forex pipeline at half past every hour. If a run is still going when the next one is due, the new run waits in line rather than canceling the one already in progress, so a checkpoint is never lost to a restart.

Each run moves through four stages:

1. Fetch. The pipeline pulls recent market data for up to 50 stocks or up to 30 forex pairs and stores it in a local SQLite database. It fetches a single timeframe, daily bars, split and dividend adjusted. Earlier versions mixed daily, hourly, and weekly bars in one table, which poisoned the prediction target; one consistent timeframe means one consistent next-day target.
2. Train. The pipeline downloads the current checkpoint from Hugging Face and continues training it for up to 2000 optimizer steps. The step count is the stop condition, which keeps every run predictable in length. Windows are built per symbol and then sorted by date, the feature scaler is fit on the training split only, and targets are clipped to a realistic daily range so a single bad bar cannot distort the objective. Training uses gradient accumulation to simulate a batch size of 256, light data augmentation, and an exponential moving average of the weights for a smoother final model.
3. Track. Every optimizer step reports its loss, learning rate, gradient norm, and elapsed time to Comet ML, so you can watch the training curves live. Per epoch metrics such as validation loss and direction accuracy are recorded as well, along with a per-symbol audit of exactly which data fed the run.
4. Gate. Before anything is published, a sanity gate runs the fresh checkpoint over held-out windows and checks for the failure signatures of a broken model: a constant output, a collapse onto one direction, or a blown-up prediction scale. If the model is degenerate it is deleted instead of pushed, and the run is marked as failed so a tracking issue opens automatically.
5. Deploy. Once a model clears the gate, the updated checkpoint is written to disk and uploaded back to Hugging Face, ready for the next run and for anyone who wants to download it.

The whole pipeline runs as a single GitHub Actions job. The model file never leaves the machine that produced it, which removes a class of failures where a checkpoint could go missing while it was handed between separate jobs.

### Training techniques

| Technique | Detail |
|-----------|--------|
| Learning rate warmup | A short linear ramp before cosine annealing begins |
| Cosine warm restarts | Periodic restarts of the learning rate to escape shallow minima |
| EMA weight averaging | A decay of 0.999, which produces weights that tend to generalize better |
| Gradient clipping | A maximum norm of 1.0 to keep gradients from exploding |
| Gradient accumulation | Builds an effective batch size of 256 from smaller micro batches |
| Data augmentation | Small Gaussian noise plus occasional masking of timesteps |
| BalancedDirectionLoss | Roughly 60 percent Huber regression and 40 percent weighted direction loss |
| Early stopping | Stops when the moving average validation loss stops improving |
| Feature clamping | Bounds features to a fixed range after normalization |
| Mixed precision | bfloat16 on CPU and float16 on CUDA |

## Technical Indicators

The model reads 44 features computed from raw open, high, low, close, and volume data. There is no zero padding, so every feature carries real information.

| Category | Indicators |
|----------|------------|
| Price | Returns, Log Returns, Volatility, ATR |
| Trend | SMA (5, 10, 20, 50, 200), EMA (5, 10, 20, 50, 200) |
| Momentum | RSI, Fast RSI, Stochastic RSI, Momentum, Rate of Change, Williams %R |
| Oscillators | MACD, MACD Signal, MACD Histogram, Stochastic K and D, CCI |
| Volatility | Bollinger Bands (Upper, Lower, Width, %B), Keltner Channels (Upper, Lower, %K) |
| Volume | Volume SMA, Volume Ratio, OBV (normalized) |
| Trend strength | ADX, Plus DI, Minus DI, Price versus SMA50 and SMA200, ATR percent |
| Mean reversion | Z Score over 20 days, Distance from the 52 week high |

## Checkpoint Format

Every checkpoint is a single `.pt` file that holds both the weights and enough configuration to rebuild the model exactly. The main keys are:

| Key | Type | Description |
|-----|------|-------------|
| `model_state_dict` | `dict` | The PyTorch model weights |
| `model_type` | `str` | Either `stock` or `forex` |
| `architecture` | `str` | `MeridianModel-2026` |
| `version` | `str` | The model version, currently `6.0.1` |
| `input_size` | `int` | `44`, the feature count |
| `seq_len` | `int` | `30`, the lookback window |
| `dim` | `int` | Hidden dimension |
| `num_layers` | `int` | Network depth |
| `num_heads` | `int` | Attention heads |
| `num_kv_heads` | `int` | Key and value heads for Grouped Query Attention |
| `num_experts` | `int` | Number of experts in the Mixture of Experts layer |
| `num_prediction_heads` | `int` | Number of output heads |
| `dropout` | `float` | Dropout rate |
| `use_mamba` | `bool` | Whether the Mamba block is active |
| `mamba_state_dim` | `int` | Size of the Mamba hidden state |
| `scaler_mean` | `Tensor` | Mean used to normalize features |
| `scaler_std` | `Tensor` | Standard deviation used to normalize features |
| `metadata` | `dict` | Best validation loss, direction accuracy, target bounds, and training history |

The loader only accepts checkpoints at version 6.0 or newer. The v6.0 architecture is deliberately smaller than the v5.x networks, so their state dictionaries are incompatible; older checkpoints are skipped, and the previous v5.x models are archived under `legacy/` on Hugging Face for reference only.

## Project Structure

```
meridianalgo/
  meridian_model.py        The MeridianModel architecture (GQA, MoE, optional Mamba SSM)
  large_torch_model.py     Training loop, data handling, checkpoint save and load, inference
  direction_loss.py        BalancedDirectionLoss and the direction accuracy metrics
  unified_ml.py            Stock prediction interface and feature engineering
  forex_ml.py              Forex prediction interface
  utils.py                 GPU detection and accuracy tracking helpers
  __init__.py              Package metadata and entry points
scripts/
  train_stocks.py          Stock training entry point
  train_forex.py           Forex training entry point
  fetch_and_store_data.py  Market data ingestion into SQLite
  push_to_hf.py            Uploads checkpoints to Hugging Face
  sanity_check_model.py    Post-training gate that blocks degenerate models from publishing
  migrate_hf_legacy.py     Moves older checkpoints into a legacy folder on Hugging Face
  clean_workflow_runs.py   Maintenance helper for old GitHub Actions runs
.github/workflows/
  stocks.yml               Hourly stock training at minute 00
  forex.yml                Hourly forex training at minute 30
  lint.yml                 Formatting and lint checks
tests/
  conftest.py                     Shared fixtures, including model checkpoints
  test_checkpoint_health.py       Checks on checkpoint metadata
  test_model_inference.py         Forward pass and state dictionary tests
  test_directional_signal.py      Direction accuracy on real market data
  test_predict_denormalization.py Verifies predictions are scaled back correctly
```

## Version History

The full record of changes lives in [docs/CHANGELOG.md](docs/CHANGELOG.md), kept separate from this document.

## Disclaimer

This software is for research and educational purposes only. It is not financial advice.

Trading financial instruments carries significant risk. Every prediction is a probabilistic forecast based on historical data, and past performance does not guarantee future results. Markets can behave in ways no model expects during sudden shocks, liquidity crises, or structural shifts.

You should never trade with money you cannot afford to lose. Any trading decision you make is yours alone. MeridianAlgo and its contributors are not liable for any financial loss that results from using this software.

The software is provided as is, without warranty of any kind. By using it you agree to hold MeridianAlgo and all contributors harmless from any claim that arises from your use of it. You are responsible for following all financial regulations that apply to you.

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.

Made with care by [MeridianAlgo](https://github.com/MeridianAlgo)
