# Multi-Currency Support

This module provides comprehensive multi-currency support for ARA AI, including real-time currency conversion, currency risk analysis, and currency-hedged return calculations.

## Features

- **Real-time Exchange Rates**: Fetch current exchange rates for 10+ major currencies
- **Currency Conversion**: Convert amounts between any supported currencies
- **Currency Risk Analysis**: Analyze currency risk in multi-currency portfolios
- **Currency-Hedged Returns**: Calculate hedged vs unhedged returns
- **User Preferences**: Store and manage user currency preferences
- **Caching**: Intelligent caching of exchange rates to minimize API calls

## Supported Currencies

- USD - US Dollar
- EUR - Euro
- GBP - British Pound
- JPY - Japanese Yen
- CNY - Chinese Yuan
- AUD - Australian Dollar
- CAD - Canadian Dollar
- CHF - Swiss Franc
- HKD - Hong Kong Dollar
- SGD - Singapore Dollar
- INR - Indian Rupee
- KRW - South Korean Won

## Quick Start

### Basic Currency Conversion

```python
import asyncio
from ara.currency import CurrencyConverter, Currency

async def main():
    converter = CurrencyConverter()
    
    # Convert 1000 USD to EUR
    result = await converter.convert(
        amount=1000,
        from_currency=Currency.USD,
        to_currency=Currency.EUR
    )
    
    print(f"${result.original_amount} USD = €{result.converted_amount:.2f} EUR")
    print(f"Exchange rate: {result.exchange_rate:.4f}")

asyncio.run(main())
```

### Get Exchange Rates

```python
import asyncio
from ara.currency import CurrencyConverter, Currency

async def main():
    converter = CurrencyConverter()
    
    # Get USD/EUR exchange rate
    rate = await converter.get_exchange_rate(
        from_currency=Currency.USD,
        to_currency=Currency.EUR
    )
    
    print(f"1 USD = {rate.rate:.4f} EUR")
    print(f"Bid: {rate.bid}, Ask: {rate.ask}")

asyncio.run(main())
```

### Currency Risk Analysis

```python
import asyncio
from ara.currency import CurrencyRiskAnalyzer, Currency

async def main():
    analyzer = CurrencyRiskAnalyzer()
    
    # Define portfolio positions
    positions = {
        'AAPL': {'amount': 10000, 'currency': Currency.USD},
        'BMW.DE': {'amount': 5000, 'currency': Currency.EUR},
        'HSBA.L': {'amount': 3000, 'currency': Currency.GBP}
    }
    
    # Analyze currency risk
    risk_metrics = await analyzer.analyze_portfolio_currency_risk(
        positions=positions,
        base_currency=Currency.USD
    )
    
    print(f"Total currency risk: ${risk_metrics.total_currency_risk:.2f}")
    print(f"Hedged return: {risk_metrics.hedged_return:.2%}")
    print(f"Unhedged return: {risk_metrics.unhedged_return:.2%}")
    print(f"Currency contribution: {risk_metrics.currency_contribution:.2%}")

asyncio.run(main())
```

### Currency Preferences

```python
from ara.currency import CurrencyPreferenceManager, Currency

# Create preference manager
pref_manager = CurrencyPreferenceManager()

# Set preferred currency
pref_manager.set_preferred_currency(Currency.EUR)

# Get preferred currency
preferred = pref_manager.get_preferred_currency()
print(f"Preferred currency: {preferred.value}")

# Configure auto-conversion
pref_manager.set_auto_convert(True)
pref_manager.set_show_original(True)
```

## API Reference

### CurrencyConverter

Main class for currency conversion operations.

#### Methods

- `get_exchange_rate(from_currency, to_currency, use_cache=True)`: Get exchange rate
- `convert(amount, from_currency, to_currency, use_cache=True)`: Convert amount
- `convert_multiple(amounts, target_currency, use_cache=True)`: Convert multiple amounts
- `get_cross_rates(base_currency, target_currencies, use_cache=True)`: Get multiple rates
- `clear_cache()`: Clear exchange rate cache
- `get_cache_stats()`: Get cache statistics

### CurrencyRiskAnalyzer

Analyze currency risk for portfolios.

#### Methods

- `calculate_currency_volatility(from_currency, to_currency, period='1y')`: Calculate volatility
- `calculate_currency_correlation(currency1, currency2, base_currency, period='1y')`: Calculate correlation
- `calculate_hedged_return(asset_return, asset_currency, base_currency, period='1y')`: Calculate hedged returns
- `analyze_portfolio_currency_risk(positions, base_currency, period='1y')`: Full portfolio analysis
- `calculate_optimal_hedge_ratio(asset_currency, base_currency, period='1y')`: Calculate hedge ratio

### CurrencyPreferenceManager

Manage user currency preferences.

#### Methods

- `load_preference()`: Load saved preference
- `save_preference(preference)`: Save preference
- `set_preferred_currency(currency)`: Set preferred currency
- `get_preferred_currency()`: Get preferred currency
- `set_auto_convert(auto_convert)`: Enable/disable auto-conversion
- `set_show_original(show_original)`: Show/hide original currency
- `reset_to_default()`: Reset to default (USD)

## Data Models

### Currency (Enum)

Enumeration of supported currencies.

### CurrencyPreference

User preference settings:
- `preferred_currency`: Preferred display currency
- `auto_convert`: Automatically convert to preferred currency
- `show_original`: Show original currency alongside converted

### ConversionResult

Result of currency conversion:
- `original_amount`: Original amount
- `original_currency`: Original currency
- `converted_amount`: Converted amount
- `target_currency`: Target currency
- `exchange_rate`: Exchange rate used
- `timestamp`: Conversion timestamp
- `source`: Data source

### ExchangeRate

Exchange rate data:
- `from_currency`: Source currency
- `to_currency`: Target currency
- `rate`: Exchange rate
- `timestamp`: Rate timestamp
- `source`: Data source
- `bid`: Bid price (optional)
- `ask`: Ask price (optional)

### CurrencyRiskMetrics

Portfolio currency risk metrics:
- `base_currency`: Base currency for analysis
- `currency_exposures`: Exposure by currency
- `currency_volatilities`: Volatility by currency
- `currency_correlations`: Currency correlations
- `total_currency_risk`: Total currency VaR
- `hedged_return`: Currency-hedged return
- `unhedged_return`: Unhedged return
- `currency_contribution`: Currency contribution to return

## Caching

The currency converter uses intelligent caching to minimize API calls:

- **Cache TTL**: 5 minutes (configurable)
- **Cache Strategy**: In-memory dictionary cache
- **Cache Invalidation**: Time-based expiration
- **Cache Statistics**: Available via `get_cache_stats()`

## Error Handling

All methods raise `DataProviderError` on failures:

```python
from ara.core.exceptions import DataProviderError

try:
    result = await converter.convert(100, Currency.USD, Currency.EUR)
except DataProviderError as e:
    print(f"Conversion failed: {e}")
```

## Integration with Predictions

The currency module integrates seamlessly with prediction results:

```python
from ara.api.prediction_engine import PredictionEngine
from ara.currency import CurrencyConverter, Currency

async def predict_with_currency():
    # Get prediction
    engine = PredictionEngine()
    prediction = await engine.predict("AAPL", days=5)
    
    # Convert predicted prices to EUR
    converter = CurrencyConverter()
    
    for daily_pred in prediction.predictions:
        result = await converter.convert(
            amount=daily_pred.predicted_price,
            from_currency=Currency.USD,
            to_currency=Currency.EUR
        )
        print(f"Day {daily_pred.day}: €{result.converted_amount:.2f}")
```

## Performance Considerations

- Exchange rates are cached for 5 minutes by default
- Batch operations use `asyncio.gather()` for parallel execution
- Historical data is fetched asynchronously
- Cache statistics help monitor performance

## Requirements

This module requires:
- `yfinance`: For fetching exchange rates
- `pandas`: For data manipulation
- `numpy`: For numerical calculations

All requirements are included in the main `requirements.txt`.

## Testing

Run tests with:

```bash
pytest tests/test_currency.py -v
```

## Future Enhancements

- Support for more currencies
- Integration with additional data providers
- Advanced hedging strategies
- Real-time streaming of exchange rates
- Historical exchange rate database
- Currency forward rates
- Options-based hedging
