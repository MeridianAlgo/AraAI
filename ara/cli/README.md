# ARA AI Enhanced CLI

The ARA AI Enhanced Command-Line Interface provides comprehensive commands for financial predictions, backtesting, portfolio management, and market analysis.

## Installation

The CLI is automatically installed with the ARA AI package:

```bash
pip install ara-ai
```

## Quick Start

```bash
# Predict stock price
ara predict AAPL --days 7

# Predict cryptocurrency
ara crypto predict BTC --days 7 --onchain

# Run backtest
ara backtest MSFT --period 2y --plot

# Optimize portfolio
ara portfolio optimize AAPL MSFT GOOGL --risk-tolerance moderate

# Check market regime
ara market regime AAPL

# Analyze sentiment
ara market sentiment BTC --sources twitter --sources reddit
```

## Command Groups

### 1. Predict (Stock/Forex)

Predict stock or forex prices with advanced analysis.

```bash
ara predict SYMBOL [OPTIONS]

Options:
  --days, -d INTEGER              Number of days to predict (default: 5)
  --analysis [basic|full|minimal] Analysis level (default: basic)
  --no-cache                      Skip cache and generate fresh predictions
  --export, -e PATH               Export results to file (JSON/CSV)
  --confidence-threshold, -c FLOAT Minimum confidence threshold (0-1)
```

**Examples:**

```bash
# Basic prediction
ara predict AAPL --days 7

# Full analysis with export
ara predict MSFT --analysis full --export results.json

# High confidence predictions only
ara predict GOOGL --confidence-threshold 0.75
```

### 2. Crypto

Cryptocurrency-specific predictions and analysis.

```bash
ara crypto COMMAND [OPTIONS]

Commands:
  predict   Predict cryptocurrency prices
  list      List supported cryptocurrencies
  compare   Compare multiple cryptocurrencies
```

**Examples:**

```bash
# Predict with on-chain metrics
ara crypto predict BTC --days 7 --onchain --defi

# Compare cryptocurrencies
ara crypto compare BTC ETH BNB --metric volatility

# List top 100 cryptocurrencies
ara crypto list --top 100
```

### 3. Backtest

Backtest prediction accuracy on historical data.

```bash
ara backtest SYMBOL [OPTIONS]

Options:
  --start, -s TEXT                Start date (YYYY-MM-DD)
  --end, -e TEXT                  End date (YYYY-MM-DD)
  --period, -p TEXT               Period (1y, 2y, 5y) (default: 1y)
  --strategy [buy_hold|momentum|mean_reversion]
  --initial-capital FLOAT         Initial capital (default: 10000.0)
  --report, -r PATH               Save detailed report to file
  --plot                          Generate equity curve plot
```

**Examples:**

```bash
# Backtest with default settings
ara backtest AAPL --period 2y

# Custom date range with plot
ara backtest MSFT --start 2020-01-01 --end 2023-12-31 --plot

# Save detailed report
ara backtest BTC --period 5y --report backtest_report.html
```

### 4. Portfolio

Portfolio optimization and analysis commands.

```bash
ara portfolio COMMAND [OPTIONS]

Commands:
  optimize   Optimize portfolio allocation
  analyze    Analyze portfolio composition and risk
  rebalance  Calculate rebalancing trades
```

**Examples:**

```bash
# Optimize portfolio
ara portfolio optimize AAPL MSFT GOOGL --risk-tolerance moderate

# Analyze with correlations
ara portfolio analyze AAPL MSFT GOOGL --show-correlations --risk-metrics

# Calculate rebalancing trades
ara portfolio rebalance AAPL MSFT GOOGL \
  --current-weights 0.4,0.3,0.3 \
  --target-weights 0.33,0.33,0.34 \
  --capital 50000
```

### 5. Models

Model management and training commands.

```bash
ara models COMMAND [OPTIONS]

Commands:
  list      List available models
  train     Train a new model
  compare   Compare model performance
  deploy    Deploy a model to production
  delete    Delete a model
```

**Examples:**

```bash
# List all models
ara models list --show-metrics

# Train a new model
ara models train AAPL --model-type transformer --epochs 200 --validate

# Compare models
ara models compare AAPL --metric accuracy

# Deploy model
ara models deploy model_abc123
```

### 6. Market

Market analysis and monitoring commands.

```bash
ara market COMMAND [OPTIONS]

Commands:
  regime        Detect current market regime
  sentiment     Analyze market sentiment
  correlations  Analyze asset correlations
  indicators    Calculate technical indicators
```

**Examples:**

```bash
# Check market regime
ara market regime AAPL --history

# Analyze sentiment
ara market sentiment BTC --sources twitter --sources reddit --timeframe 7d

# Calculate correlations
ara market correlations AAPL MSFT GOOGL --window 90 --heatmap

# Show technical indicators
ara market indicators AAPL --category trend
```

### 7. Config

Configuration management commands.

```bash
ara config COMMAND [OPTIONS]

Commands:
  init      Initialize ARA AI configuration
  show      Show current configuration
  set       Set configuration value
  get       Get configuration value
  validate  Validate configuration
  reset     Reset configuration to defaults
```

**Examples:**

```bash
# Interactive configuration wizard
ara config init --interactive

# Show configuration
ara config show

# Set configuration value
ara config set cache.enabled true

# Get configuration value
ara config get data_providers.primary

# Validate configuration
ara config validate
```

## Interactive Features

### Progress Bars

Long-running operations display progress bars:

```bash
ara backtest AAPL --period 5y
# Shows: [████████████████████] 100% Running backtest for AAPL...
```

### Rich Terminal Output

The CLI uses Rich library for beautiful terminal output:
- Color-coded results
- Formatted tables
- Progress indicators
- Syntax highlighting

### Interactive Prompts

Configuration wizard provides interactive prompts:

```bash
ara config init --interactive
# Prompts for:
# - Data provider selection
# - API keys
# - Cache settings
# - Model preferences
```

## Output Formats

### JSON Export

```bash
ara predict AAPL --export results.json
```

### CSV Export

```bash
ara predict AAPL --export results.csv
```

### HTML Reports

```bash
ara backtest AAPL --report backtest_report.html
```

### Interactive Plots

```bash
ara backtest AAPL --plot
# Generates: backtest_aapl_equity_curve.html

ara market correlations AAPL MSFT GOOGL --heatmap
# Generates: correlation_heatmap.html
```

## Environment Variables

Configure CLI behavior with environment variables:

```bash
# Enable debug mode
export ARA_DEBUG=true

# Set configuration path
export ARA_CONFIG_PATH=/path/to/config.yaml

# Set cache directory
export ARA_CACHE_DIR=/path/to/cache
```

## Shell Completion

Generate shell completion scripts:

```bash
# Bash
ara --install-completion bash

# Zsh
ara --install-completion zsh

# Fish
ara --install-completion fish
```

## Error Handling

The CLI provides helpful error messages and suggestions:

```bash
ara predict INVALID_SYMBOL
# Output:
# ✗ Data Provider Error: Symbol not found
# Tip: Check your internet connection and API keys
```

## Tips and Best Practices

1. **Use caching for faster results:**
   ```bash
   ara predict AAPL  # Uses cache if available
   ```

2. **Force fresh predictions:**
   ```bash
   ara predict AAPL --no-cache
   ```

3. **Export results for analysis:**
   ```bash
   ara predict AAPL --export results.json
   ```

4. **Combine commands with shell pipes:**
   ```bash
   ara predict AAPL --export - | jq '.predictions[0]'
   ```

5. **Use confidence thresholds for quality:**
   ```bash
   ara predict AAPL --confidence-threshold 0.75
   ```

## Troubleshooting

### Command not found

```bash
# Reinstall package
pip install --upgrade ara-ai

# Or use python -m
python -m ara.cli predict AAPL
```

### Import errors

```bash
# Install missing dependencies
pip install -r requirements.txt
```

### Configuration issues

```bash
# Reset configuration
ara config reset --force

# Reinitialize
ara config init --interactive
```

## Advanced Usage

### Batch Processing

```bash
# Process multiple symbols
for symbol in AAPL MSFT GOOGL; do
  ara predict $symbol --export ${symbol}_results.json
done
```

### Automated Backtesting

```bash
# Backtest multiple strategies
for strategy in buy_hold momentum mean_reversion; do
  ara backtest AAPL --strategy $strategy --report backtest_${strategy}.html
done
```

### Portfolio Monitoring

```bash
# Daily portfolio analysis
ara portfolio analyze AAPL MSFT GOOGL --risk-metrics > daily_report.txt
```

## API Integration

The CLI commands can be used programmatically:

```python
from ara.cli import main
import sys

# Simulate CLI call
sys.argv = ['ara', 'predict', 'AAPL', '--days', '7']
main()
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/MeridianAlgo/AraAI/issues
- Documentation: https://github.com/MeridianAlgo/AraAI/docs
- Email: support@meridianalgo.com
