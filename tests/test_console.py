"""
Unit tests for Console Manager
"""
import pytest
from meridianalgo.console import ConsoleManager


class TestConsoleManager:
    """Test suite for Console Manager"""
    
    @pytest.fixture
    def console(self):
        """Create console manager instance"""
        return ConsoleManager(verbose=False)
    
    def test_initialization(self, console):
        """Test console initialization"""
        assert console is not None
        assert hasattr(console, 'console')
        
    def test_verbose_mode(self):
        """Test verbose mode"""
        console_verbose = ConsoleManager(verbose=True)
        assert console_verbose.verbose is True
        
        console_quiet = ConsoleManager(verbose=False)
        assert console_quiet.verbose is False
        
    def test_print_methods(self, console):
        """Test various print methods"""
        # These should not raise exceptions
        console.print_success("Test success")
        console.print_error("Test error")
        console.print_warning("Test warning")
        console.print_info("Test info")
        
    def test_ultimate_predictions_display(self, console):
        """Test ultimate predictions display"""
        # Mock result
        result = {
            'symbol': 'AAPL',
            'current_price': 150.0,
            'predictions': [
                {
                    'day': 1,
                    'date': '2025-09-22',
                    'predicted_price': 151.0,
                    'predicted_return': 0.0067,
                    'confidence': 0.95
                }
            ],
            'model_accuracy': 98.5,
            'model_type': 'ultimate_ensemble_8_models',
            'feature_count': 44,
            'market_status': {'is_open': False},
            'sector_info': {
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'market_cap': 2500000000000
            },
            'hf_sentiment': {
                'label': 'POSITIVE',
                'score': 0.85
            },
            'financial_health': {
                'health_score': 75,
                'health_grade': 'B+',
                'risk_grade': 'Low-Moderate Risk',
                'debt_to_equity': 1.5,
                'current_ratio': 1.2
            }
        }
        
        # Should not raise exception
        console.print_ultimate_predictions(result)
        
    def test_header_display(self, console):
        """Test header display"""
        console.print_header("Test Header")
        # Should not raise exception


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
