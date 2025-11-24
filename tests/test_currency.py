"""
Tests for multi-currency support
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from ara.currency import (
    CurrencyConverter,
    CurrencyRiskAnalyzer,
    CurrencyPreferenceManager,
    Currency,
    CurrencyPreference,
    ConversionResult,
    ExchangeRate,
    CurrencyRiskMetrics
)
from ara.core.exceptions import DataProviderError


class TestCurrencyModels:
    """Test currency data models"""
    
    def test_currency_enum(self):
        """Test Currency enum"""
        assert Currency.USD.value == "USD"
        assert Currency.EUR.value == "EUR"
        assert Currency.GBP.value == "GBP"
        assert len(Currency) >= 10  # At least 10 currencies
    
    def test_currency_preference(self):
        """Test CurrencyPreference model"""
        pref = CurrencyPreference(
            preferred_currency=Currency.EUR,
            auto_convert=True,
            show_original=False
        )
        
        assert pref.preferred_currency == Currency.EUR
        assert pref.auto_convert is True
        assert pref.show_original is False
        
        # Test to_dict
        data = pref.to_dict()
        assert data['preferred_currency'] == 'EUR'
        assert data['auto_convert'] is True
        
        # Test from_dict
        pref2 = CurrencyPreference.from_dict(data)
        assert pref2.preferred_currency == Currency.EUR
        assert pref2.auto_convert is True
    
    def test_conversion_result(self):
        """Test ConversionResult model"""
        result = ConversionResult(
            original_amount=100.0,
            original_currency=Currency.USD,
            converted_amount=85.0,
            target_currency=Currency.EUR,
            exchange_rate=0.85,
            timestamp=datetime.now(),
            source="test"
        )
        
        assert result.original_amount == 100.0
        assert result.converted_amount == 85.0
        assert result.exchange_rate == 0.85
        
        # Test to_dict
        data = result.to_dict()
        assert data['original_currency'] == 'USD'
        assert data['target_currency'] == 'EUR'


class TestCurrencyConverter:
    """Test CurrencyConverter"""
    
    @pytest.mark.asyncio
    async def test_identity_conversion(self):
        """Test conversion between same currency"""
        converter = CurrencyConverter()
        
        result = await converter.convert(
            amount=100,
            from_currency=Currency.USD,
            to_currency=Currency.USD
        )
        
        assert result.original_amount == 100
        assert result.converted_amount == 100
        assert result.exchange_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_identity_exchange_rate(self):
        """Test exchange rate for same currency"""
        converter = CurrencyConverter()
        
        rate = await converter.get_exchange_rate(
            from_currency=Currency.USD,
            to_currency=Currency.USD
        )
        
        assert rate.rate == 1.0
        assert rate.from_currency == Currency.USD
        assert rate.to_currency == Currency.USD
    
    @pytest.mark.asyncio
    async def test_real_conversion(self):
        """Test real currency conversion"""
        converter = CurrencyConverter()
        
        try:
            result = await converter.convert(
                amount=100,
                from_currency=Currency.USD,
                to_currency=Currency.EUR
            )
            
            assert result.original_amount == 100
            assert result.converted_amount > 0
            assert result.exchange_rate > 0
            assert result.original_currency == Currency.USD
            assert result.target_currency == Currency.EUR
            
        except DataProviderError:
            pytest.skip("Exchange rate API unavailable")
    
    @pytest.mark.asyncio
    async def test_caching(self):
        """Test exchange rate caching"""
        converter = CurrencyConverter(cache_ttl=60)
        
        try:
            # First call - should fetch
            rate1 = await converter.get_exchange_rate(
                Currency.USD,
                Currency.EUR
            )
            
            # Second call - should use cache
            rate2 = await converter.get_exchange_rate(
                Currency.USD,
                Currency.EUR
            )
            
            assert rate1.rate == rate2.rate
            
            # Check cache stats
            stats = converter.get_cache_stats()
            assert stats['total_entries'] > 0
            
        except DataProviderError:
            pytest.skip("Exchange rate API unavailable")
    
    @pytest.mark.asyncio
    async def test_convert_multiple(self):
        """Test batch conversion"""
        converter = CurrencyConverter()
        
        amounts = {
            Currency.USD: 100,
            Currency.EUR: 100
        }
        
        try:
            results = await converter.convert_multiple(
                amounts=amounts,
                target_currency=Currency.GBP
            )
            
            assert len(results) == 2
            assert Currency.USD in results
            assert Currency.EUR in results
            
            for result in results.values():
                assert result.converted_amount > 0
                assert result.target_currency == Currency.GBP
                
        except DataProviderError:
            pytest.skip("Exchange rate API unavailable")
    
    @pytest.mark.asyncio
    async def test_get_cross_rates(self):
        """Test getting cross rates"""
        converter = CurrencyConverter()
        
        target_currencies = [Currency.EUR, Currency.GBP, Currency.JPY]
        
        try:
            rates = await converter.get_cross_rates(
                base_currency=Currency.USD,
                target_currencies=target_currencies
            )
            
            assert len(rates) == 3
            
            for currency in target_currencies:
                assert currency in rates
                assert rates[currency].rate > 0
                
        except DataProviderError:
            pytest.skip("Exchange rate API unavailable")
    
    def test_clear_cache(self):
        """Test cache clearing"""
        converter = CurrencyConverter()
        
        # Add something to cache manually
        converter._rate_cache[(Currency.USD, Currency.EUR)] = ExchangeRate(
            from_currency=Currency.USD,
            to_currency=Currency.EUR,
            rate=0.85,
            timestamp=datetime.now(),
            source="test"
        )
        
        assert len(converter._rate_cache) > 0
        
        converter.clear_cache()
        
        assert len(converter._rate_cache) == 0


class TestCurrencyRiskAnalyzer:
    """Test CurrencyRiskAnalyzer"""
    
    @pytest.mark.asyncio
    async def test_currency_volatility(self):
        """Test currency volatility calculation"""
        analyzer = CurrencyRiskAnalyzer()
        
        try:
            volatility = await analyzer.calculate_currency_volatility(
                from_currency=Currency.EUR,
                to_currency=Currency.USD,
                period="1y"
            )
            
            assert volatility >= 0
            assert volatility < 1.0  # Reasonable volatility range
            
        except DataProviderError:
            pytest.skip("Currency data unavailable")
    
    @pytest.mark.asyncio
    async def test_currency_correlation(self):
        """Test currency correlation calculation"""
        analyzer = CurrencyRiskAnalyzer()
        
        try:
            correlation = await analyzer.calculate_currency_correlation(
                currency1=Currency.EUR,
                currency2=Currency.GBP,
                base_currency=Currency.USD,
                period="1y"
            )
            
            assert -1 <= correlation <= 1
            
        except DataProviderError:
            pytest.skip("Currency data unavailable")
    
    @pytest.mark.asyncio
    async def test_hedged_return_same_currency(self):
        """Test hedged return for same currency"""
        analyzer = CurrencyRiskAnalyzer()
        
        hedged, unhedged = await analyzer.calculate_hedged_return(
            asset_return=0.10,
            asset_currency=Currency.USD,
            base_currency=Currency.USD
        )
        
        assert hedged == 0.10
        assert unhedged == 0.10
    
    @pytest.mark.asyncio
    async def test_hedged_return_different_currency(self):
        """Test hedged return for different currency"""
        analyzer = CurrencyRiskAnalyzer()
        
        try:
            hedged, unhedged = await analyzer.calculate_hedged_return(
                asset_return=0.10,
                asset_currency=Currency.EUR,
                base_currency=Currency.USD,
                period="1y"
            )
            
            assert hedged == 0.10  # Hedged return is asset return only
            assert isinstance(unhedged, float)
            
        except DataProviderError:
            pytest.skip("Currency data unavailable")
    
    @pytest.mark.asyncio
    async def test_portfolio_currency_risk(self):
        """Test portfolio currency risk analysis"""
        analyzer = CurrencyRiskAnalyzer()
        
        positions = {
            'AAPL': {'amount': 10000, 'currency': Currency.USD},
            'BMW': {'amount': 5000, 'currency': Currency.EUR}
        }
        
        try:
            risk_metrics = await analyzer.analyze_portfolio_currency_risk(
                positions=positions,
                base_currency=Currency.USD
            )
            
            assert risk_metrics.base_currency == Currency.USD
            assert len(risk_metrics.currency_exposures) == 2
            assert Currency.USD in risk_metrics.currency_exposures
            assert Currency.EUR in risk_metrics.currency_exposures
            assert risk_metrics.total_currency_risk >= 0
            
        except DataProviderError:
            pytest.skip("Currency data unavailable")
    
    @pytest.mark.asyncio
    async def test_optimal_hedge_ratio(self):
        """Test optimal hedge ratio calculation"""
        analyzer = CurrencyRiskAnalyzer()
        
        # Same currency - no hedging needed
        ratio = await analyzer.calculate_optimal_hedge_ratio(
            asset_currency=Currency.USD,
            base_currency=Currency.USD
        )
        
        assert ratio == 0.0
        
        # Different currency
        try:
            ratio = await analyzer.calculate_optimal_hedge_ratio(
                asset_currency=Currency.EUR,
                base_currency=Currency.USD,
                period="1y"
            )
            
            assert 0 <= ratio <= 1
            
        except DataProviderError:
            pytest.skip("Currency data unavailable")


class TestCurrencyPreferenceManager:
    """Test CurrencyPreferenceManager"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.pref_manager = CurrencyPreferenceManager(config_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_preference(self):
        """Test default preference"""
        pref = self.pref_manager.load_preference()
        
        assert pref.preferred_currency == Currency.USD
        assert pref.auto_convert is True
        assert pref.show_original is True
    
    def test_save_and_load_preference(self):
        """Test saving and loading preference"""
        # Create preference
        pref = CurrencyPreference(
            preferred_currency=Currency.EUR,
            auto_convert=False,
            show_original=True
        )
        
        # Save
        self.pref_manager.save_preference(pref)
        
        # Load
        loaded_pref = self.pref_manager.load_preference()
        
        assert loaded_pref.preferred_currency == Currency.EUR
        assert loaded_pref.auto_convert is False
        assert loaded_pref.show_original is True
    
    def test_set_preferred_currency(self):
        """Test setting preferred currency"""
        self.pref_manager.set_preferred_currency(Currency.GBP)
        
        pref = self.pref_manager.load_preference()
        assert pref.preferred_currency == Currency.GBP
    
    def test_get_preferred_currency(self):
        """Test getting preferred currency"""
        self.pref_manager.set_preferred_currency(Currency.JPY)
        
        currency = self.pref_manager.get_preferred_currency()
        assert currency == Currency.JPY
    
    def test_set_auto_convert(self):
        """Test setting auto-convert flag"""
        self.pref_manager.set_auto_convert(False)
        
        pref = self.pref_manager.load_preference()
        assert pref.auto_convert is False
    
    def test_set_show_original(self):
        """Test setting show-original flag"""
        self.pref_manager.set_show_original(False)
        
        pref = self.pref_manager.load_preference()
        assert pref.show_original is False
    
    def test_reset_to_default(self):
        """Test resetting to default"""
        # Set non-default values
        self.pref_manager.set_preferred_currency(Currency.EUR)
        self.pref_manager.set_auto_convert(False)
        
        # Reset
        self.pref_manager.reset_to_default()
        
        # Check
        pref = self.pref_manager.load_preference()
        assert pref.preferred_currency == Currency.USD
        assert pref.auto_convert is True
        assert pref.show_original is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
