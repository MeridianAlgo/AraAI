"""
Smoke tests for ARA AI - Quick validation tests for CI/CD.
These tests should complete quickly and verify basic functionality.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImports:
    """Test that all modules can be imported."""
    
    def test_import_package(self):
        """Test main package import."""
        import meridianalgo
        assert meridianalgo.__version__ == '3.0.0'
    
    def test_import_ultimate_ml(self):
        """Test Ultimate ML import."""
        from meridianalgo.ultimate_ml import UltimateStockML
        assert UltimateStockML is not None
    
    def test_import_console(self):
        """Test Console import."""
        from meridianalgo.console import ConsoleManager
        assert ConsoleManager is not None
    
    def test_import_data(self):
        """Test Data Manager import."""
        from meridianalgo.data import DataManager
        assert DataManager is not None
    
    def test_import_core(self):
        """Test Core import."""
        from meridianalgo.core import AraAI
        assert AraAI is not None


class TestBasicFunctionality:
    """Test basic functionality without network calls."""
    
    def test_ml_initialization(self):
        """Test ML system can be initialized."""
        from meridianalgo.ultimate_ml import UltimateStockML
        
        ml = UltimateStockML()
        assert ml is not None
        assert hasattr(ml, 'models')
        assert hasattr(ml, 'scalers')
    
    def test_console_initialization(self):
        """Test console can be initialized."""
        from meridianalgo.console import ConsoleManager
        
        console = ConsoleManager()
        assert console is not None
    
    def test_data_manager_initialization(self):
        """Test data manager can be initialized."""
        from meridianalgo.data import DataManager
        
        dm = DataManager()
        assert dm is not None
    
    def test_model_status(self):
        """Test model status retrieval."""
        from meridianalgo.ultimate_ml import UltimateStockML
        
        ml = UltimateStockML()
        status = ml.get_model_status()
        
        assert isinstance(status, dict)
        assert 'is_trained' in status
        assert 'models' in status
        assert 'feature_count' in status
    
    def test_market_status(self):
        """Test market status detection."""
        from meridianalgo.ultimate_ml import UltimateStockML
        
        ml = UltimateStockML()
        status = ml.get_market_status()
        
        assert isinstance(status, dict)
        assert 'is_open' in status
    
    def test_sector_detection(self):
        """Test sector detection."""
        from meridianalgo.ultimate_ml import UltimateStockML
        
        ml = UltimateStockML()
        sector = ml.get_stock_sector('AAPL')
        
        assert isinstance(sector, dict)
        assert 'sector' in sector
        assert sector['sector'] == 'Technology'


class TestConsoleOutput:
    """Test console output methods."""
    
    def test_console_methods(self):
        """Test all console output methods."""
        from meridianalgo.console import ConsoleManager
        
        console = ConsoleManager()
        
        # These should not raise exceptions
        console.print_header("Test Header")
        console.print_success("Success message")
        console.print_warning("Warning message")
        console.print_error("Error message")
        console.print_info("Info message")
        
        assert True


class TestPathHandling:
    """Test path handling across platforms."""
    
    def test_model_directory(self):
        """Test model directory creation."""
        from meridianalgo.ultimate_ml import UltimateStockML
        from pathlib import Path
        
        ml = UltimateStockML()
        assert isinstance(ml.model_dir, Path)
        assert ml.model_dir.exists()
    
    def test_cache_directory(self):
        """Test cache directory handling."""
        from pathlib import Path
        
        cache_dir = Path.home() / '.araai' / 'cache'
        # Just verify Path works, don't require directory exists
        assert isinstance(cache_dir, Path)


class TestVersioning:
    """Test version information."""
    
    def test_package_version(self):
        """Test package version."""
        import meridianalgo
        
        version = meridianalgo.__version__
        assert version == '3.0.0'
        assert isinstance(version, str)
    
    def test_version_format(self):
        """Test version format."""
        import meridianalgo
        
        version = meridianalgo.__version__
        parts = version.split('.')
        
        # Should be semantic versioning
        assert len(parts) >= 2
        assert parts[0].isdigit()
        assert parts[1].isdigit()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
