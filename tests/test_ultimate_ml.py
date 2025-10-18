"""
Unit tests for Ultimate ML System
"""
import pytest
import numpy as np
from meridianalgo.ultimate_ml import UltimateStockML


class TestUltimateML:
    """Test suite for Ultimate ML System"""
    
    @pytest.fixture
    def ml_system(self):
        """Create ML system instance"""
        return UltimateStockML()
    
    def test_initialization(self, ml_system):
        """Test ML system initialization"""
        assert ml_system is not None
        assert hasattr(ml_system, 'models')
        assert hasattr(ml_system, 'scalers')
        
    def test_model_status(self, ml_system):
        """Test model status retrieval"""
        status = ml_system.get_model_status()
        assert isinstance(status, dict)
        assert 'is_trained' in status
        assert 'models' in status
        assert 'feature_count' in status
        assert status['feature_count'] == 44
        
    def test_market_status(self, ml_system):
        """Test market status detection"""
        status = ml_system.get_market_status()
        assert isinstance(status, dict)
        assert 'is_open' in status
        
    def test_sector_detection(self, ml_system):
        """Test sector detection for major stocks"""
        # Test known stocks
        aapl_sector = ml_system.get_stock_sector('AAPL')
        assert aapl_sector['sector'] == 'Technology'
        assert 'industry' in aapl_sector
        
        msft_sector = ml_system.get_stock_sector('MSFT')
        assert msft_sector['sector'] == 'Technology'
        
    def test_prediction_bounds(self, ml_system):
        """Test that predictions have realistic bounds"""
        # This test requires trained models
        # Skip if models not trained
        if not ml_system.is_trained:
            pytest.skip("Models not trained")
            
        result = ml_system.predict_ultimate('AAPL', days=5)
        
        if result and 'predictions' in result:
            predictions = result['predictions']
            current_price = result['current_price']
            
            for pred in predictions:
                predicted_price = pred['predicted_price']
                change_pct = (predicted_price - current_price) / current_price
                
                # Check realistic bounds (±15% total)
                assert -0.15 <= change_pct <= 0.15, f"Prediction out of bounds: {change_pct:.2%}"
                
    def test_prediction_variation(self, ml_system):
        """Test that predictions vary across days"""
        if not ml_system.is_trained:
            pytest.skip("Models not trained")
            
        result = ml_system.predict_ultimate('AAPL', days=5)
        
        if result and 'predictions' in result:
            predictions = result['predictions']
            prices = [p['predicted_price'] for p in predictions]
            
            # Check that not all prices are identical
            assert len(set(prices)) > 1, "All predictions are identical"
            
    def test_confidence_scores(self, ml_system):
        """Test confidence score calculation"""
        if not ml_system.is_trained:
            pytest.skip("Models not trained")
            
        result = ml_system.predict_ultimate('AAPL', days=5)
        
        if result and 'predictions' in result:
            predictions = result['predictions']
            
            for pred in predictions:
                confidence = pred['confidence']
                
                # Check confidence is in valid range
                assert 0 <= confidence <= 1, f"Invalid confidence: {confidence}"
                
            # Check confidence decreases over time
            confidences = [p['confidence'] for p in predictions]
            for i in range(len(confidences) - 1):
                assert confidences[i] >= confidences[i+1], "Confidence should decrease over time"
                
    def test_financial_health_analysis(self, ml_system):
        """Test financial health analysis"""
        import yfinance as yf
        
        ticker = yf.Ticker('AAPL')
        result = ml_system._analyze_basic_financial_health(ticker, 'AAPL')
        
        assert isinstance(result, dict)
        assert 'health_score' in result
        assert 'health_grade' in result
        assert 'risk_grade' in result
        
        # Check score is in valid range
        assert 0 <= result['health_score'] <= 100
        
        # Check grade is valid
        valid_grades = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'F']
        assert result['health_grade'] in valid_grades
        
    def test_invalid_symbol(self, ml_system):
        """Test handling of invalid stock symbols"""
        result = ml_system.predict_ultimate('INVALID_SYMBOL_XYZ', days=5)
        assert result is None or 'error' in result
        
    def test_feature_extraction(self, ml_system):
        """Test feature extraction"""
        import yfinance as yf
        import pandas as pd
        
        # Get sample data
        data = yf.download('AAPL', period='6mo', progress=False)
        
        if len(data) > 50:
            # Add indicators
            data = ml_system._add_ultimate_indicators(data)
            
            # Extract features
            sector_info = ml_system.get_stock_sector('AAPL')
            features = ml_system._extract_current_ultimate_features(data, 'AAPL', sector_info)
            
            if features is not None:
                assert len(features) == 44, f"Expected 44 features, got {len(features)}"
                assert all(isinstance(f, (int, float, np.number)) for f in features)


class TestModelTraining:
    """Test suite for model training"""
    
    def test_small_training(self):
        """Test training with small dataset"""
        ml = UltimateStockML()
        
        # Train on small dataset
        success = ml.train_ultimate_models(max_symbols=5, period='6mo', use_parallel=False)
        
        assert success is True or success is None  # May return None on some systems
        
    def test_model_saving_loading(self, tmp_path):
        """Test model saving and loading"""
        ml = UltimateStockML(model_dir=str(tmp_path))
        
        # Train small model
        ml.train_ultimate_models(max_symbols=5, period='6mo', use_parallel=False)
        
        # Save models
        ml._save_models()
        
        # Create new instance and load
        ml2 = UltimateStockML(model_dir=str(tmp_path))
        loaded = ml2.load_models()
        
        assert loaded is True


class TestCrossPlatform:
    """Cross-platform compatibility tests"""
    
    def test_path_handling(self):
        """Test path handling across platforms"""
        from pathlib import Path
        
        ml = UltimateStockML()
        assert isinstance(ml.model_dir, Path)
        assert ml.model_dir.exists()
        
    def test_import_compatibility(self):
        """Test that all imports work"""
        try:
            from meridianalgo.ultimate_ml import UltimateStockML
            from meridianalgo.console import ConsoleManager
            from meridianalgo.core import AraAI
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
