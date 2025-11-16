"""
Tests for Backward Compatibility Layer

Tests the compatibility wrappers, migration tools, and deprecation system.
"""

import pytest
import warnings
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from ara.compat import (
    AraAI,
    StockPredictor,
    predict_stock,
    analyze_stock,
    ModelMigrator,
    DataMigrator,
    deprecated,
    DeprecationLevel
)
from ara.compat.deprecation import DeprecationTimeline, get_warning_manager


class TestDeprecationSystem:
    """Test deprecation warning system"""
    
    def test_deprecated_decorator(self):
        """Test deprecated decorator issues warnings"""
        
        @deprecated(
            reason="Test deprecation",
            version="4.0.0",
            removal_version="5.0.0",
            alternative="new_function()",
            level=DeprecationLevel.WARNING
        )
        def old_function():
            return "result"
        
        # Should issue warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_function()
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert result == "result"
    
    def test_deprecation_metadata(self):
        """Test deprecated functions have metadata"""
        
        @deprecated(
            reason="Test",
            version="4.0.0",
            alternative="new_func()"
        )
        def old_func():
            pass
        
        assert hasattr(old_func, '__deprecated__')
        assert old_func.__deprecated__ is True
        assert hasattr(old_func, '__deprecation_info__')
        assert old_func.__deprecation_info__['reason'] == "Test"
        assert old_func.__deprecation_info__['version'] == "4.0.0"
    
    def test_deprecation_timeline(self):
        """Test deprecation timeline constants"""
        assert DeprecationTimeline.get_current_version() == "4.0.0"
        assert DeprecationTimeline.get_removal_version() == "5.0.0"
        assert isinstance(DeprecationTimeline.get_removal_date(), datetime)
    
    def test_warning_manager(self):
        """Test deprecation warning manager"""
        manager = get_warning_manager()
        
        # Test warn_once
        manager.warn_once("test_feature", "Test warning")
        
        # Get summary
        summary = manager.get_deprecation_summary()
        assert isinstance(summary, dict)
        assert "warnings_issued" in summary
        assert "current_version" in summary
        assert "removal_version" in summary


class TestCompatibilityWrappers:
    """Test API compatibility wrappers"""
    
    def test_predict_stock_function(self):
        """Test predict_stock convenience function"""
        # Should issue deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Note: This will fail without actual data/models
            # but we're testing the wrapper exists and issues warnings
            try:
                result = predict_stock('AAPL', days=5, verbose=False)
            except Exception:
                pass  # Expected to fail without data
            
            # Check warning was issued
            assert len(w) > 0
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
    
    def test_analyze_stock_function(self):
        """Test analyze_stock convenience function"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = analyze_stock('AAPL', verbose=False)
            
            # Should return message about redesign
            assert isinstance(result, dict)
            assert "message" in result
            
            # Check warning was issued
            assert len(w) > 0
    
    def test_araai_class_initialization(self):
        """Test AraAI class can be initialized"""
        # Reset warning manager to ensure warnings are issued
        from ara.compat.deprecation import get_warning_manager
        manager = get_warning_manager()
        manager._warnings_issued.clear()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            ara = AraAI(verbose=False)
            
            assert ara is not None
            assert hasattr(ara, 'predict')
            assert hasattr(ara, '_engine')
            
            # Check warning was issued (either through warnings or warning manager)
            # The warning manager may suppress duplicate warnings
            assert len(w) > 0 or len(manager._warnings_issued) > 0
    
    def test_stock_predictor_class(self):
        """Test StockPredictor class"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            predictor = StockPredictor(verbose=False)
            
            assert predictor is not None
            assert hasattr(predictor, 'predict')
            assert hasattr(predictor, 'ara')
            
            # Check warning was issued
            assert len(w) > 0
    
    def test_format_conversion(self):
        """Test response format conversion"""
        ara = AraAI(verbose=False)
        
        # Create mock new format response
        new_format = {
            "symbol": "AAPL",
            "asset_type": "stock",
            "current_price": 150.0,
            "predictions": [
                {
                    "day": 1,
                    "date": datetime(2024, 1, 2),
                    "predicted_price": 152.0,
                    "predicted_return": 1.33,
                    "confidence": 0.85,
                    "lower_bound": 148.0,
                    "upper_bound": 156.0
                }
            ],
            "confidence": {
                "overall": 0.85,
                "model_agreement": 0.90
            },
            "explanations": {
                "top_factors": [],
                "natural_language": "Test"
            },
            "regime": {
                "current_regime": "bull",
                "confidence": 0.85
            },
            "timestamp": datetime.now(),
            "model_version": "4.0.0"
        }
        
        # Convert to old format
        old_format = ara._convert_to_old_format(new_format)
        
        # Verify old format structure
        assert "symbol" in old_format
        assert "current_price" in old_format
        assert "predictions" in old_format
        assert "timestamp" in old_format
        
        # Verify prediction format
        pred = old_format["predictions"][0]
        assert "day" in pred
        assert "predicted_price" in pred
        assert "change" in pred
        assert "change_pct" in pred


class TestModelMigration:
    """Test model migration tools"""
    
    def test_model_migrator_initialization(self):
        """Test ModelMigrator can be initialized"""
        migrator = ModelMigrator(verbose=False)
        assert migrator is not None
        assert hasattr(migrator, 'migrate_model_directory')
        assert hasattr(migrator, 'migrate_model_file')
    
    def test_migrate_nonexistent_directory(self):
        """Test migration of nonexistent directory"""
        migrator = ModelMigrator(verbose=False)
        
        with pytest.raises(FileNotFoundError):
            migrator.migrate_model_directory(
                old_dir=Path('/nonexistent/path'),
                new_dir=Path('/tmp/new'),
                backup=False
            )
    
    def test_migrate_empty_directory(self):
        """Test migration of empty directory"""
        migrator = ModelMigrator(verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir = Path(tmpdir) / 'old'
            new_dir = Path(tmpdir) / 'new'
            old_dir.mkdir()
            
            summary = migrator.migrate_model_directory(
                old_dir=old_dir,
                new_dir=new_dir,
                backup=False
            )
            
            assert summary['total_files'] == 0
            assert summary['migrated'] == 0
            assert summary['failed'] == 0
    
    def test_migrate_json_config(self):
        """Test migration of JSON configuration"""
        migrator = ModelMigrator(verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir = Path(tmpdir) / 'old'
            new_dir = Path(tmpdir) / 'new'
            old_dir.mkdir()
            new_dir.mkdir()
            
            # Create test JSON file
            test_config = {
                "version": "3.1.1",
                "model_path": "models/",
                "cache_dir": ".ara_cache/"
            }
            
            config_file = old_dir / 'config.json'
            with open(config_file, 'w') as f:
                json.dump(test_config, f)
            
            # Migrate
            result = migrator.migrate_model_file(config_file, new_dir)
            
            assert result['success']
            assert Path(result['new_file']).exists()
            
            # Verify migrated config
            with open(result['new_file'], 'r') as f:
                new_config = json.load(f)
            
            assert new_config['version'] == "4.0.0"
            assert 'migrated_from' in new_config
    
    def test_get_migration_log(self):
        """Test getting migration log"""
        migrator = ModelMigrator(verbose=False)
        
        log = migrator.get_migration_log()
        assert isinstance(log, list)


class TestDataMigration:
    """Test data migration tools"""
    
    def test_data_migrator_initialization(self):
        """Test DataMigrator can be initialized"""
        migrator = DataMigrator(verbose=False)
        assert migrator is not None
        assert hasattr(migrator, 'migrate_cache_directory')
        assert hasattr(migrator, 'validate_migration')
    
    def test_migrate_nonexistent_cache(self):
        """Test migration of nonexistent cache"""
        migrator = DataMigrator(verbose=False)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = migrator.migrate_cache_directory(
                old_cache_dir=Path('/nonexistent/cache'),
                new_cache_dir=Path('/tmp/new_cache'),
                backup=False
            )
            
            assert not result.get('success', True)
    
    def test_migrate_cache_file(self):
        """Test migration of cache file"""
        migrator = DataMigrator(verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir = Path(tmpdir) / 'old_cache'
            new_dir = Path(tmpdir) / 'new_cache'
            old_dir.mkdir()
            new_dir.mkdir()
            
            # Create test cache file
            test_cache = {
                "symbol": "AAPL",
                "predictions": [
                    {
                        "day": 1,
                        "predicted_price": 150.0
                    }
                ],
                "timestamp": "2024-01-01T00:00:00"
            }
            
            cache_file = old_dir / 'AAPL_cache.json'
            with open(cache_file, 'w') as f:
                json.dump(test_cache, f)
            
            # Migrate
            result = migrator._migrate_cache_file(cache_file, new_dir)
            
            assert result['success']
            assert Path(result['new_file']).exists()
            
            # Verify migrated cache
            with open(result['new_file'], 'r') as f:
                new_cache = json.load(f)
            
            assert 'cache_version' in new_cache
            assert new_cache['cache_version'] == "4.0.0"
    
    def test_validate_migration(self):
        """Test migration validation"""
        migrator = DataMigrator(verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'test'
            test_dir.mkdir()
            
            # Create valid migrated file
            test_data = {
                "cache_version": "4.0.0",
                "data": "test"
            }
            
            test_file = test_dir / 'test.json'
            with open(test_file, 'w') as f:
                json.dump(test_data, f)
            
            # Validate
            result = migrator.validate_migration(test_dir)
            
            assert 'valid' in result
            assert 'total_files' in result
            assert result['total_files'] > 0


class TestIntegration:
    """Integration tests for compatibility layer"""
    
    def test_end_to_end_compatibility(self):
        """Test end-to-end compatibility workflow"""
        # This test verifies the complete compatibility workflow
        
        # Reset warning manager
        from ara.compat.deprecation import get_warning_manager
        manager = get_warning_manager()
        manager._warnings_issued.clear()
        
        # 1. Initialize with old API
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            ara = AraAI(verbose=False)
            
            # Should issue warning (either through warnings or warning manager)
            assert len(w) > 0 or len(manager._warnings_issued) > 0
        
        # 2. Check system info
        info = ara.get_system_info()
        assert info['compatibility_mode'] is True
        assert 'version' in info
        
        # 3. Test deprecated methods exist
        assert hasattr(ara, 'predict')
        assert hasattr(ara, 'predict_with_ai')
        assert hasattr(ara, 'analyze_accuracy')
        assert hasattr(ara, 'validate_predictions')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
