"""
Configuration management system
Loads settings from YAML files and environment variables
Supports multiple environments (dev, staging, prod)
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from ara.core.exceptions import ConfigurationError


class Environment(Enum):
    """Supported environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    
    @classmethod
    def from_string(cls, env_str: str) -> 'Environment':
        """Convert string to Environment enum"""
        env_map = {
            'dev': cls.DEVELOPMENT,
            'development': cls.DEVELOPMENT,
            'stage': cls.STAGING,
            'staging': cls.STAGING,
            'prod': cls.PRODUCTION,
            'production': cls.PRODUCTION
        }
        env_lower = env_str.lower()
        if env_lower not in env_map:
            raise ConfigurationError(
                f"Invalid environment: {env_str}",
                {"valid_environments": list(env_map.keys())}
            )
        return env_map[env_lower]


@dataclass
class DataConfig:
    """Data provider configuration"""
    cache_ttl: int = 300  # seconds
    max_retries: int = 3
    timeout: int = 30
    providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate data configuration"""
        if self.cache_ttl < 0:
            raise ConfigurationError("data.cache_ttl must be non-negative")
        if self.cache_ttl > 86400:  # 24 hours
            raise ConfigurationError("data.cache_ttl must not exceed 86400 seconds (24 hours)")
        
        if self.max_retries < 1:
            raise ConfigurationError("data.max_retries must be at least 1")
        if self.max_retries > 10:
            raise ConfigurationError("data.max_retries must not exceed 10")
        
        if self.timeout < 1:
            raise ConfigurationError("data.timeout must be at least 1 second")
        if self.timeout > 300:
            raise ConfigurationError("data.timeout must not exceed 300 seconds")
        
        # Validate provider configurations
        for provider_name, provider_config in self.providers.items():
            if not isinstance(provider_config, dict):
                raise ConfigurationError(
                    f"Provider '{provider_name}' configuration must be a dictionary"
                )
            if 'enabled' in provider_config and not isinstance(provider_config['enabled'], bool):
                raise ConfigurationError(
                    f"Provider '{provider_name}' enabled flag must be boolean"
                )


@dataclass
class ModelConfig:
    """Model configuration"""
    model_dir: str = "models"
    default_ensemble_size: int = 12
    gpu_enabled: bool = True
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 0.0005
    
    def validate(self) -> None:
        """Validate model configuration"""
        if self.default_ensemble_size < 1:
            raise ConfigurationError("model.default_ensemble_size must be at least 1")
        if self.default_ensemble_size > 50:
            raise ConfigurationError("model.default_ensemble_size must not exceed 50")
        
        if self.batch_size < 1:
            raise ConfigurationError("model.batch_size must be at least 1")
        if self.batch_size > 1024:
            raise ConfigurationError("model.batch_size must not exceed 1024")
        
        if self.epochs < 1:
            raise ConfigurationError("model.epochs must be at least 1")
        if self.epochs > 10000:
            raise ConfigurationError("model.epochs must not exceed 10000")
        
        if self.learning_rate <= 0:
            raise ConfigurationError("model.learning_rate must be positive")
        if self.learning_rate > 1.0:
            raise ConfigurationError("model.learning_rate must not exceed 1.0")
        
        # Validate model directory
        if not self.model_dir:
            raise ConfigurationError("model.model_dir must not be empty")


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    rate_limit: int = 100  # requests per minute
    enable_cors: bool = True
    
    def validate(self) -> None:
        """Validate API configuration"""
        if self.port < 1 or self.port > 65535:
            raise ConfigurationError("api.port must be between 1 and 65535")
        
        if self.workers < 1:
            raise ConfigurationError("api.workers must be at least 1")
        if self.workers > 128:
            raise ConfigurationError("api.workers must not exceed 128")
        
        if self.rate_limit < 1:
            raise ConfigurationError("api.rate_limit must be at least 1")
        if self.rate_limit > 100000:
            raise ConfigurationError("api.rate_limit must not exceed 100000")
        
        # Validate host format (basic check)
        if not self.host:
            raise ConfigurationError("api.host must not be empty")


@dataclass
class CacheConfig:
    """Cache configuration"""
    enabled: bool = True
    redis_url: Optional[str] = None
    ttl_l1: int = 10  # seconds
    ttl_l2: int = 300  # seconds
    max_size_mb: int = 1000
    
    def validate(self) -> None:
        """Validate cache configuration"""
        if self.ttl_l1 < 0:
            raise ConfigurationError("cache.ttl_l1 must be non-negative")
        if self.ttl_l1 > 3600:
            raise ConfigurationError("cache.ttl_l1 must not exceed 3600 seconds")
        
        if self.ttl_l2 < 0:
            raise ConfigurationError("cache.ttl_l2 must be non-negative")
        if self.ttl_l2 > 86400:
            raise ConfigurationError("cache.ttl_l2 must not exceed 86400 seconds")
        
        if self.ttl_l1 > self.ttl_l2:
            raise ConfigurationError("cache.ttl_l1 must not exceed cache.ttl_l2")
        
        if self.max_size_mb < 1:
            raise ConfigurationError("cache.max_size_mb must be at least 1")
        if self.max_size_mb > 100000:
            raise ConfigurationError("cache.max_size_mb must not exceed 100000")
        
        # Validate Redis URL format if provided
        if self.redis_url and not self.redis_url.startswith(('redis://', 'rediss://')):
            raise ConfigurationError(
                "cache.redis_url must start with 'redis://' or 'rediss://'"
            )


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"
    file: Optional[str] = None
    console: bool = True
    
    def validate(self) -> None:
        """Validate logging configuration"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.level.upper() not in valid_levels:
            raise ConfigurationError(
                f"Invalid log level: {self.level}",
                {"valid_levels": valid_levels}
            )
        
        valid_formats = ['json', 'text', 'structured']
        if self.format.lower() not in valid_formats:
            raise ConfigurationError(
                f"Invalid log format: {self.format}",
                {"valid_formats": valid_formats}
            )
        
        # Validate log file path if provided
        if self.file:
            log_path = Path(self.file)
            if log_path.exists() and not log_path.is_file():
                raise ConfigurationError(
                    f"Log file path exists but is not a file: {self.file}"
                )


class Config:
    """
    Main configuration class
    Loads and validates configuration from multiple sources
    
    Supports multiple environments:
    - development: Local development with verbose logging
    - staging: Pre-production testing environment
    - production: Production environment with optimized settings
    
    Configuration priority (highest to lowest):
    1. Environment variables (ARA_*)
    2. YAML configuration file
    3. Default values
    """
    
    def __init__(self, config_path: Optional[Path] = None, env: str = "development"):
        # Validate and set environment
        self.env_enum = Environment.from_string(env)
        self.env = self.env_enum.value
        self.config_path = config_path or self._get_default_config_path()
        
        # Initialize sub-configs with defaults
        self.data = DataConfig()
        self.model = ModelConfig()
        self.api = APIConfig()
        self.cache = CacheConfig()
        self.logging = LoggingConfig()
        
        # Load configuration in priority order
        self._load_config()
        self._load_env_overrides()
        self._validate()
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path"""
        # Look for config in multiple locations
        locations = [
            Path("ara/config/config.yaml"),
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path.home() / ".ara" / "config.yaml"
        ]
        
        for loc in locations:
            if loc.exists():
                return loc
        
        # Return default location (may not exist yet)
        return Path("ara/config/config.yaml")
    
    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            print(f"Config file not found: {self.config_path}, using defaults")
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                return
            
            # Load environment-specific config
            env_config = config_data.get(self.env, {})
            
            # Update data config
            if 'data' in env_config:
                self._update_dataclass(self.data, env_config['data'])
            
            # Update model config
            if 'model' in env_config:
                self._update_dataclass(self.model, env_config['model'])
            
            # Update API config
            if 'api' in env_config:
                self._update_dataclass(self.api, env_config['api'])
            
            # Update cache config
            if 'cache' in env_config:
                self._update_dataclass(self.cache, env_config['cache'])
            
            # Update logging config
            if 'logging' in env_config:
                self._update_dataclass(self.logging, env_config['logging'])
                
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load config from {self.config_path}",
                {"error": str(e)}
            )
    
    def _update_dataclass(self, obj: Any, data: Dict[str, Any]) -> None:
        """Update dataclass fields from dictionary"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def _load_env_overrides(self) -> None:
        """Load configuration overrides from environment variables"""
        # Data config
        if os.getenv('ARA_CACHE_TTL'):
            self.data.cache_ttl = int(os.getenv('ARA_CACHE_TTL'))
        
        # Model config
        if os.getenv('ARA_MODEL_DIR'):
            self.model.model_dir = os.getenv('ARA_MODEL_DIR')
        if os.getenv('ARA_GPU_ENABLED'):
            self.model.gpu_enabled = os.getenv('ARA_GPU_ENABLED').lower() == 'true'
        
        # API config
        if os.getenv('ARA_API_HOST'):
            self.api.host = os.getenv('ARA_API_HOST')
        if os.getenv('ARA_API_PORT'):
            self.api.port = int(os.getenv('ARA_API_PORT'))
        
        # Cache config
        if os.getenv('ARA_REDIS_URL'):
            self.cache.redis_url = os.getenv('ARA_REDIS_URL')
        
        # Logging config
        if os.getenv('ARA_LOG_LEVEL'):
            self.logging.level = os.getenv('ARA_LOG_LEVEL')
    
    def _validate(self) -> None:
        """Validate all configuration values"""
        try:
            self.data.validate()
            self.model.validate()
            self.api.validate()
            self.cache.validate()
            self.logging.validate()
        except ConfigurationError as e:
            # Re-raise with environment context
            context = {"environment": self.env, "original_error": str(e)}
            if hasattr(e, 'context') and e.context:
                context.update(e.context)
            raise ConfigurationError(
                f"Configuration validation failed for environment '{self.env}': {str(e)}",
                context
            )
        
        # Cross-config validation
        self._validate_cross_config()
    
    def _validate_cross_config(self) -> None:
        """Validate relationships between different config sections"""
        # Ensure model directory exists or can be created
        model_path = Path(self.model.model_dir)
        if model_path.exists() and not model_path.is_dir():
            raise ConfigurationError(
                f"Model directory path exists but is not a directory: {self.model.model_dir}"
            )
        
        # Warn if GPU is enabled but might not be available
        if self.model.gpu_enabled:
            try:
                import torch
                if not torch.cuda.is_available():
                    print(f"Warning: GPU enabled in config but CUDA not available")
            except ImportError:
                print(f"Warning: GPU enabled but PyTorch not installed")
        
        # Validate cache configuration consistency
        if self.cache.enabled and self.cache.redis_url:
            # Redis cache enabled - ensure data cache_ttl is reasonable
            if self.data.cache_ttl > self.cache.ttl_l2:
                print(f"Warning: data.cache_ttl ({self.data.cache_ttl}s) exceeds cache.ttl_l2 ({self.cache.ttl_l2}s)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "env": self.env,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "api": self.api.__dict__,
            "cache": self.cache.__dict__,
            "logging": self.logging.__dict__
        }
    
    def save(self, path: Optional[Path] = None) -> None:
        """
        Save configuration to YAML file
        
        Args:
            path: Optional path to save config. Uses config_path if not provided.
        """
        save_path = path or self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing config to preserve other environments
        existing_config = {}
        if save_path.exists():
            try:
                with open(save_path, 'r') as f:
                    existing_config = yaml.safe_load(f) or {}
            except Exception:
                pass  # If load fails, start fresh
        
        # Update only the current environment
        existing_config[self.env] = {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "api": self.api.__dict__,
            "cache": self.cache.__dict__,
            "logging": self.logging.__dict__
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(existing_config, f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def create_default_config(path: Path, environments: Optional[List[str]] = None) -> None:
        """
        Create a default configuration file with all environments
        
        Args:
            path: Path where to create the config file
            environments: List of environments to include. Defaults to all.
        """
        if environments is None:
            environments = [e.value for e in Environment]
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {}
        for env in environments:
            # Create config for each environment with appropriate defaults
            temp_config = Config(env=env)
            config_data[env] = {
                "data": temp_config.data.__dict__,
                "model": temp_config.model.__dict__,
                "api": temp_config.api.__dict__,
                "cache": temp_config.cache.__dict__,
                "logging": temp_config.logging.__dict__
            }
            
            # Adjust defaults based on environment
            if env == Environment.PRODUCTION.value:
                config_data[env]["api"]["workers"] = 8
                config_data[env]["api"]["enable_cors"] = False
                config_data[env]["cache"]["redis_url"] = "redis://localhost:6379/0"
                config_data[env]["logging"]["level"] = "WARNING"
                config_data[env]["logging"]["console"] = False
            elif env == Environment.STAGING.value:
                config_data[env]["api"]["workers"] = 4
                config_data[env]["cache"]["redis_url"] = "redis://localhost:6379/1"
                config_data[env]["logging"]["level"] = "INFO"
        
        with open(path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    
    def get_environment_name(self) -> str:
        """Get the current environment name"""
        return self.env
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.env_enum == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.env_enum == Environment.DEVELOPMENT
    
    def is_staging(self) -> bool:
        """Check if running in staging environment"""
        return self.env_enum == Environment.STAGING


# Global configuration instance
_config: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    Get global configuration instance
    
    Args:
        reload: Force reload configuration
        
    Returns:
        Config instance
    """
    global _config
    
    if _config is None or reload:
        env = os.getenv('ARA_ENV', 'development')
        _config = Config(env=env)
    
    return _config
