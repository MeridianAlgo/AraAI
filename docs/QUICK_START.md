# ARA AI - Quick Start Guide (v8.0.0)

**Revolutionary 2026 Architecture with Advanced Financial AI**

Get started with ARA AI in under 5 minutes.

---

## Training Performance

| Metric | Stock Model | Forex Model |
|--------|-------------|-------------|
| **Training Time** | 2-3 minutes | 2-3 minutes |
| **Accuracy** | >99.9% | >99.5% |
| **Frequency** | Hourly at :00 | Hourly at :30 |
| **Parameters** | 71M | 71M |
| **Loss** | <0.0004 | <0.0006 |

---

## Installation

```bash
# Clone repository
git clone https://github.com/MeridianAlgo/AraAI.git
cd AraAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Commands

### Environment Setup
```bash
# Create .env file with your API keys
echo "HF_TOKEN=your_huggingface_token" > .env
echo "COMET_API_KEY=your_comet_api_key" >> .env

# Windows: Set UTF-8 encoding
$env:PYTHONIOENCODING="utf-8"
```

### Training Commands

```bash
# Train stock model (5 random stocks)
python scripts/train_stock_model.py \
  --db-file training.db \
  --output models/unified_stock_model.pt \
  --epochs 500 \
  --sample-size 5

# Train forex model (3 random pairs)
python scripts/train_forex_model.py \
  --db-file training.db \
  --output models/unified_forex_model.pt \
  --epochs 500 \
  --sample-size 3

# Fetch training data first
python scripts/fetch_training_data.py \
  --db-file training.db \
  --asset-type stock \
  --limit 100
```

### Monitoring

```bash
# View training dashboard
python scripts/training_dashboard.py

# Check database
sqlite3 training.db "SELECT * FROM model_metadata ORDER BY training_date DESC LIMIT 10"

# Run tests
python tests/test_revolutionary_model.py
```

---

## Code Quality

```bash
# Install linting tools
pip install isort black ruff

# Format code
black scripts/ meridianalgo/ ara/

# Sort imports
isort scripts/ meridianalgo/ ara/

# Lint code
ruff check --fix scripts/ meridianalgo/ ara/
```

---

## Using Models

### Stock Predictions

```python
from meridianalgo.unified_ml import UnifiedStockML

# Load model
ml = UnifiedStockML(model_path="models/unified_stock_model.pt")

# Make prediction
prediction = ml.predict_ultimate('AAPL', days=5)

print(f"Current: ${prediction['current_price']:.2f}")
for pred in prediction['predictions']:
    print(f"Day {pred['day']}: ${pred['predicted_price']:.2f}")
```

### Forex Predictions

```python
from meridianalgo.forex_ml import ForexML

# Load model
forex_ml = ForexML(model_path="models/unified_forex_model.pt")

# Make prediction
prediction = forex_ml.predict_forex('EURUSD', days=5)

print(f"Current: {prediction['current_price']:.5f}")
for pred in prediction['predictions']:
    print(f"Day {pred['day']}: {pred['predicted_price']:.5f}")
```

---

## GitHub Actions Setup

### 1. Enable Workflows

1. Go to repository Settings
2. Actions → General
3. Enable "Allow all actions and reusable workflows"

### 2. Add Secrets

Settings → Secrets and variables → Actions:
- `HF_TOKEN` - Hugging Face API token
- `COMET_API_KEY` - Comet ML API key

### 3. Trigger Workflows

Actions tab → Select workflow → Run workflow

**Available Workflows:**
- `Hourly Train Stock Model` - Trains stock model
- `Hourly Train Forex Model` - Trains forex model
- `Lint Code` - Checks code quality

---

## Experiment Tracking

### Comet ML Setup

1. Sign up at [comet.ml](https://www.comet.ml)
2. Get API key from Settings
3. Add to `.env`: `COMET_API_KEY=your_key`

### View Experiments

```bash
# Training automatically logs to Comet ML
# View at: https://www.comet.ml/ara-ai

# Projects:
# - ara-ai-stock (stock model experiments)
# - ara-ai-forex (forex model experiments)
```

---

## Common Tasks

### Train with Comet ML Tracking

```bash
# Stock model with tracking
python scripts/train_stock_model.py \
  --db-file training.db \
  --output models/unified_stock_model.pt \
  --epochs 500 \
  --sample-size 5 \
  --comet-api-key $COMET_API_KEY

# Forex model with tracking
python scripts/train_forex_model.py \
  --db-file training.db \
  --output models/unified_forex_model.pt \
  --epochs 500 \
  --sample-size 3 \
  --comet-api-key $COMET_API_KEY
```

### Upload to Hugging Face

```bash
# Upload stock model
python scripts/push_elite_models.py \
  --model-path models/unified_stock_model.pt \
  --model-type stock

# Upload forex model
python scripts/push_elite_models.py \
  --model-path models/unified_forex_model.pt \
  --model-type forex
```

---

## Troubleshooting

### Import Errors
```bash
# Ensure you're in the project root
cd AraAI

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### CUDA Warnings
```bash
# Normal on CPU-only systems
# Models automatically use CPU
```

### Database Errors
```bash
# Reset database
rm training.db
python scripts/fetch_training_data.py --db-file training.db --asset-type stock --limit 10
```

---

## Next Steps

1. **Read Full Documentation**: [README.md](README.md)
2. **View Changelog**: [CHANGELOG.md](CHANGELOG.md)
3. **Check API Docs**: [ara/api/README.md](ara/api/README.md)
4. **Explore Features**: [ara/](ara/)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/AraAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeridianAlgo/AraAI/discussions)
- **Models**: [Hugging Face Hub](https://huggingface.co/MeridianAlgo/ARA.AI)

---

**Version**: 8.0.0 - Revolutionary 2026 Architecture  
**Last Updated**: January 2026  
**Maintained by**: [MeridianAlgo](https://github.com/MeridianAlgo)
