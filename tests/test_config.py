"""
Tests for configuration management system
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path

from ara.config import Config, Environment, get_config
from ara.core.exceptions import ConfigurationError


class TestEnvironment:
    """Test Environment enum"""
    
    def test_from_string_valid(self):
        """Test valid environment string conversion"""
        assert Environment.from_string("development") == Environment.DEVELOPMENT
        assert Environment.from_string("dev") == Environment.DEVELOPMENT
        assert Environment.from_string("staging") == Environment.STAGING
        assert Environment.from_string("stage") == Environment.STAGING
        assert Environment.from_string("production") == Environment.PRODUCTION
        assert Environment.from_string("prod") == Environment.PRODUCTION
    
    def test_from_string_case_insensitive(self):
        """Test case-insensitive environment conversion"""
        assert Environment.from_string("DEVELOPMENT") == Environment.DEVELOPMENT
        assert Environment.from_string("Dev") == Environment.DEVELOPMENT
        assert Environment.from_string("PROD") == Environment.PRODUCTION
    
    def test_from_string_invalid(self):
        """Test invalid environment string raises error"""
        with pytest.raises(ConfigurationError):
            Environment.from_string("invalid")


class TestConfig:
    """Test Config class"""
    
    def test_default_config(self):
        """Test loading default configuration"""
        config = Config(env="development")
        
        assert config.env == "development"
        assert config.data.cache_ttl == 300
        assert config.model.gpu_enabled is True
        assert config.api.port == 8000
        assert config.cache.enabled is True
        assert config.logging.level == "INFO"
    
    def test_environment_methods(self):
        """Test environment checking methods"""
        dev_config = Config(env="development")
        assert dev_config.is_development()
        assert not dev_config.is_staging()
        assert not dev_config.is_production()
        
        staging_config = Config(env="staging")
        assert not staging_config.is_development()
        assert staging_config.is_staging()
        assert not staging_config.is_production()
        
        prod_config = Config(env="production")
        assert not prod_config.is_development()
        assert not prod_config.is_staging()
        assert prod_config.is_production()
    
    def test_validation_cache_ttl(self):
        """Test cache_ttl validation"""
        config = Config(env="development")
        
        # Valid values
        config.data.cache_ttl = 0
        config._validate()
        
        config.data.cache_ttl = 86400
        config._validate()
        
        # Invalid values
        config.data.cache_ttl = -1
        with pytest.raises(ConfigurationError):
            config._validate()
        
        config.data.cache_ttl = 86401
        with pytest.raises(ConfigurationError):
            config._validate()
    
    def test_validation_port(self):
        """Test port validation"""
        config = Config(env="development")
        
        # Valid values
        config.api.port = 1
        config._validate()
        
        config.api.port = 65535
        config._validate()
        
        # Invalid values
        config.api.port = 0
        with pytest.raises(ConfigurationError):
            config._validate()
        
        config.api.port = 65536
        with pytest.raises(ConfigurationError):
            config._validate()
    
    def test_validation_log_level(self):
        """Test log level validation"""
        config = Config(env="development")
        
        # Valid values
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            config.logging.level = level
            config._validate()
        
        # Invalid value
        config.logging.level = "INVALID"
        with pytest.raises(ConfigurationError):
            config._validate()
    
    def test_validation_learning_rate(self):
        """Test learning rate validation"""
        config = Config(env="development")
        
        # Valid values
        config.model.learning_rate = 0.0001
        config._validate()
        
        config.model.learning_rate = 1.0
        config._validate()
        
        # Invalid values
        config.model.learning_rate = 0
        with pytest.raises(ConfigurationError):
            config._validate()
        
        config.model.learning_rate = 1.1
        with pytest.raises(ConfigurationError):
            config._validate()
    
    def test_to_dict(self):
        """Test configuration to dictionary conversion"""
        config = Config(env="development")
        config_dict = config.to_dict()
        
        assert config_dict["env"] == "development"
        assert "data" in config_dict
        assert "model" in config_dict
        assert "api" in config_dict
        assert "cache" in config_dict
        assert "logging" in config_dict
    
    def test_save_and_load(self):
        """Test saving and loading configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            
            # Create and save config
            config1 = Config(config_path=config_path, env="development")
            config1.api.port = 9000
            config1.logging.level = "DEBUG"
            config1.save()
            
            # Load config
            config2 = Config(config_path=config_path, env="development")
            
            assert config2.api.port == 9000
            assert config2.logging.level == "DEBUG"
    
    def test_create_default_config(self):
        """Test creating default configuration file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "default_config.yaml"
            
            Config.create_default_config(config_path)
            
            assert config_path.exists()
            
            # Load and verify
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            assert "development" in config_data
            assert "staging" in config_data
            assert "production" in config_data
    
    def test_env_override(self):
        """Test environment variable override"""
        # Set environment variables
        os.environ['ARA_API_PORT'] = '9999'
        os.environ['ARA_LOG_LEVEL'] = 'ERROR'
        os.environ['ARA_GPU_ENABLED'] = 'false'
        
        try:
            config = Config(env="development")
            
            assert config.api.port == 9999
            assert config.logging.level == "ERROR"
            assert config.model.gpu_enabled is False
        finally:
            # Clean up
            del os.environ['ARA_API_PORT']
            del os.environ['ARA_LOG_LEVEL']
            del os.environ['ARA_GPU_ENABLED']


class TestGetConfig:
    """Test get_config function"""
    
    def test_get_config_singleton(self):
        """Test that get_config returns singleton"""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_get_config_reload(self):
        """Test get_config reload"""
        config1 = get_config()
        config1.api.port = 9999
        
        config2 = get_config(reload=True)
        
        # After reload, should be back to default
        assert config2.api.port != 9999
