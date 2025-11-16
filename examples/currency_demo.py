"""
Multi-Currency Support Demo

This script demonstrates the multi-currency features of ARA AI:
- Real-time currency conversion
- Exchange rate fetching
- Currency risk analysis
- Currency preferences
- Currency-hedged returns
"""

import asyncio
from ara.currency import (
    CurrencyConverter,
    CurrencyRiskAnalyzer,
    CurrencyPreferenceManager,
    Currency
)


async def demo_basic_conversion():
    """Demonstrate basic currency conversion"""
    print("\n" + "="*60)
    print("BASIC CURRENCY CONVERSION")
    print("="*60)
    
    converter = CurrencyConverter()
    
    # Convert USD to EUR
    result = await converter.convert(
        amount=1000,
        from_currency=Currency.USD,
        to_currency=Currency.EUR
    )
    
    print(f"\nConversion: ${result.original_amount:.2f} USD -> EUR {result.converted_amount:.2f}")
    print(f"Exchange Rate: {result.exchange_rate:.4f}")
    print(f"Source: {result.source}")
    print(f"Timestamp: {result.timestamp}")
    
    # Convert to multiple currencies
    print("\n" + "-"*60)
    print("Converting $1000 to multiple currencies:")
    print("-"*60)
    
    target_currencies = [Currency.EUR, Currency.GBP, Currency.JPY, Currency.CNY]
    
    for target in target_currencies:
        result = await converter.convert(1000, Currency.USD, target)
        print(f"  {target.value}: {result.converted_amount:.2f}")


async def demo_exchange_rates():
    """Demonstrate exchange rate fetching"""
    print("\n" + "="*60)
    print("EXCHANGE RATES")
    print("="*60)
    
    converter = CurrencyConverter()
    
    # Get single exchange rate
    rate = await converter.get_exchange_rate(
        from_currency=Currency.USD,
        to_currency=Currency.EUR
    )
    
    print(f"\nUSD/EUR Exchange Rate:")
    print(f"  Rate: {rate.rate:.4f}")
    print(f"  Bid: {rate.bid}")
    print(f"  Ask: {rate.ask}")
    print(f"  Source: {rate.source}")
    
    # Get cross rates
    print("\n" + "-"*60)
    print("Cross Rates (Base: USD):")
    print("-"*60)
    
    target_currencies = [
        Currency.EUR, Currency.GBP, Currency.JPY,
        Currency.CNY, Currency.AUD, Currency.CAD
    ]
    
    cross_rates = await converter.get_cross_rates(
        base_currency=Currency.USD,
        target_currencies=target_currencies
    )
    
    for currency, rate in cross_rates.items():
        print(f"  USD/{currency.value}: {rate.rate:.4f}")
    
    # Cache statistics
    print("\n" + "-"*60)
    print("Cache Statistics:")
    print("-"*60)
    
    stats = converter.get_cache_stats()
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Valid entries: {stats['valid_entries']}")
    print(f"  Expired entries: {stats['expired_entries']}")
    print(f"  Cache TTL: {stats['cache_ttl']} seconds")


async def demo_currency_risk():
    """Demonstrate currency risk analysis"""
    print("\n" + "="*60)
    print("CURRENCY RISK ANALYSIS")
    print("="*60)
    
    analyzer = CurrencyRiskAnalyzer()
    
    # Define a multi-currency portfolio
    positions = {
        'AAPL': {'amount': 10000, 'currency': Currency.USD},
        'BMW.DE': {'amount': 5000, 'currency': Currency.EUR},
        'HSBA.L': {'amount': 3000, 'currency': Currency.GBP},
        'SONY': {'amount': 500000, 'currency': Currency.JPY}
    }
    
    print("\nPortfolio Positions:")
    print("-"*60)
    for symbol, pos in positions.items():
        print(f"  {symbol}: {pos['amount']} {pos['currency'].value}")
    
    # Analyze currency risk
    print("\nAnalyzing currency risk (this may take a moment)...")
    
    risk_metrics = await analyzer.analyze_portfolio_currency_risk(
        positions=positions,
        base_currency=Currency.USD
    )
    
    print("\n" + "-"*60)
    print("Currency Risk Metrics:")
    print("-"*60)
    
    print(f"\nBase Currency: {risk_metrics.base_currency.value}")
    
    print("\nCurrency Exposures:")
    for currency, amount in risk_metrics.currency_exposures.items():
        print(f"  {currency.value}: {amount:.2f}")
    
    print("\nCurrency Volatilities (Annualized):")
    for currency, vol in risk_metrics.currency_volatilities.items():
        print(f"  {currency.value}: {vol:.2%}")
    
    print(f"\nTotal Currency Risk (VaR): ${risk_metrics.total_currency_risk:.2f}")
    print(f"Hedged Return: {risk_metrics.hedged_return:.2%}")
    print(f"Unhedged Return: {risk_metrics.unhedged_return:.2%}")
    print(f"Currency Contribution: {risk_metrics.currency_contribution:.2%}")


async def demo_hedged_returns():
    """Demonstrate currency-hedged return calculations"""
    print("\n" + "="*60)
    print("CURRENCY-HEDGED RETURNS")
    print("="*60)
    
    analyzer = CurrencyRiskAnalyzer()
    
    # Calculate hedged vs unhedged returns
    asset_return = 0.15  # 15% return in local currency
    
    print(f"\nAsset Return (Local Currency): {asset_return:.2%}")
    print("\n" + "-"*60)
    print("Hedged vs Unhedged Returns:")
    print("-"*60)
    
    currencies = [Currency.EUR, Currency.GBP, Currency.JPY, Currency.CNY]
    
    for currency in currencies:
        hedged, unhedged = await analyzer.calculate_hedged_return(
            asset_return=asset_return,
            asset_currency=currency,
            base_currency=Currency.USD
        )
        
        currency_effect = unhedged - hedged
        
        print(f"\n{currency.value} Asset:")
        print(f"  Hedged Return: {hedged:.2%}")
        print(f"  Unhedged Return: {unhedged:.2%}")
        print(f"  Currency Effect: {currency_effect:.2%}")


def demo_preferences():
    """Demonstrate currency preference management"""
    print("\n" + "="*60)
    print("CURRENCY PREFERENCES")
    print("="*60)
    
    pref_manager = CurrencyPreferenceManager()
    
    # Get current preference
    current = pref_manager.get_preferred_currency()
    print(f"\nCurrent Preferred Currency: {current.value}")
    
    # Set new preference
    print("\nSetting preferred currency to EUR...")
    pref_manager.set_preferred_currency(Currency.EUR)
    
    # Configure auto-conversion
    print("Enabling auto-conversion...")
    pref_manager.set_auto_convert(True)
    
    print("Enabling show original...")
    pref_manager.set_show_original(True)
    
    # Load and display preference
    preference = pref_manager.load_preference()
    
    print("\n" + "-"*60)
    print("Current Preference Settings:")
    print("-"*60)
    print(f"  Preferred Currency: {preference.preferred_currency.value}")
    print(f"  Auto Convert: {preference.auto_convert}")
    print(f"  Show Original: {preference.show_original}")
    
    # Reset to default
    print("\nResetting to default (USD)...")
    pref_manager.reset_to_default()
    
    preference = pref_manager.load_preference()
    print(f"Preferred Currency: {preference.preferred_currency.value}")


async def demo_batch_conversion():
    """Demonstrate batch currency conversion"""
    print("\n" + "="*60)
    print("BATCH CURRENCY CONVERSION")
    print("="*60)
    
    converter = CurrencyConverter()
    
    # Convert multiple amounts to USD
    amounts = {
        Currency.EUR: 1000,
        Currency.GBP: 800,
        Currency.JPY: 100000,
        Currency.CNY: 5000
    }
    
    print("\nConverting multiple amounts to USD:")
    print("-"*60)
    
    for currency, amount in amounts.items():
        print(f"  {amount} {currency.value}")
    
    print("\nConverting...")
    
    results = await converter.convert_multiple(
        amounts=amounts,
        target_currency=Currency.USD
    )
    
    print("\nResults:")
    print("-"*60)
    
    total_usd = 0
    for currency, result in results.items():
        print(f"  {result.original_amount} {currency.value} = ${result.converted_amount:.2f} USD")
        total_usd += result.converted_amount
    
    print(f"\nTotal: ${total_usd:.2f} USD")


async def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("ARA AI - MULTI-CURRENCY SUPPORT DEMO")
    print("="*60)
    
    try:
        # Basic conversion
        await demo_basic_conversion()
        
        # Exchange rates
        await demo_exchange_rates()
        
        # Batch conversion
        await demo_batch_conversion()
        
        # Currency preferences
        demo_preferences()
        
        # Hedged returns
        await demo_hedged_returns()
        
        # Currency risk (may take longer)
        await demo_currency_risk()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
