"""
Configuration Management Demo

This script demonstrates how to use the ARA AI configuration system.
"""

import os
from pathlib import Path
from ara.config import Config, get_config, Environment


def demo_basic_usage():
    """Demonstrate basic configuration usage"""
    print("=" * 60)
    print("Basic Configuration Usage")
    print("=" * 60)
    
    # Get global configuration instance
    config = get_config()
    
    print(f"\nCurrent environment: {config.env}")
    print(f"Is production: {config.is_production()}")
    print(f"Is development: {config.is_development()}")
    print(f"Is staging: {config.is_staging()}")
    
    print("\nData Configuration:")
    print(f"  Cache TTL: {config.data.cache_ttl}s")
    print(f"  Max Retries: {config.data.max_retries}")
    print(f"  Timeout: {config.data.timeout}s")
    
    print("\nModel Configuration:")
    print(f"  Model Directory: {config.model.model_dir}")
    print(f"  Ensemble Size: {config.model.default_ensemble_size}")
    print(f"  GPU Enabled: {config.model.gpu_enabled}")
    print(f"  Batch Size: {config.model.batch_size}")
    
    print("\nAPI Configuration:")
    print(f"  Host: {config.api.host}")
    print(f"  Port: {config.api.port}")
    print(f"  Workers: {config.api.workers}")
    print(f"  Rate Limit: {config.api.rate_limit} req/min")
    
    print("\nCache Configuration:")
    print(f"  Enabled: {config.cache.enabled}")
    print(f"  Redis URL: {config.cache.redis_url or 'In-memory'}")
    print(f"  L1 TTL: {config.cache.ttl_l1}s")
    print(f"  L2 TTL: {config.cache.ttl_l2}s")
    
    print("\nLogging Configuration:")
    print(f"  Level: {config.logging.level}")
    print(f"  Format: {config.logging.format}")
    print(f"  Console: {config.logging.console}")


def demo_environment_switching():
    """Demonstrate switching between environments"""
    print("\n" + "=" * 60)
    print("Environment Switching")
    print("=" * 60)
    
    for env_name in ["development", "staging", "production"]:
        print(f"\n{env_name.upper()} Environment:")
        config = Config(env=env_name)
        
        print(f"  API Workers: {config.api.workers}")
        print(f"  CORS Enabled: {config.api.enable_cors}")
        print(f"  Log Level: {config.logging.level}")
        print(f"  Cache Redis: {config.cache.redis_url or 'In-memory'}")


def demo_env_overrides():
    """Demonstrate environment variable overrides"""
    print("\n" + "=" * 60)
    print("Environment Variable Overrides")
    print("=" * 60)
    
    # Set some environment variables
    os.environ['ARA_API_PORT'] = '9000'
    os.environ['ARA_LOG_LEVEL'] = 'DEBUG'
    os.environ['ARA_GPU_ENABLED'] = 'false'
    
    print("\nSet environment variables:")
    print("  ARA_API_PORT=9000")
    print("  ARA_LOG_LEVEL=DEBUG")
    print("  ARA_GPU_ENABLED=false")
    
    # Create new config (will pick up env vars)
    config = Config(env="development")
    
    print("\nConfiguration values (after override):")
    print(f"  API Port: {config.api.port}")
    print(f"  Log Level: {config.logging.level}")
    print(f"  GPU Enabled: {config.model.gpu_enabled}")
    
    # Clean up
    del os.environ['ARA_API_PORT']
    del os.environ['ARA_LOG_LEVEL']
    del os.environ['ARA_GPU_ENABLED']


def demo_custom_config():
    """Demonstrate loading custom configuration"""
    print("\n" + "=" * 60)
    print("Custom Configuration")
    print("=" * 60)
    
    # Create a temporary custom config
    import tempfile
    import yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        custom_config = {
            "development": {
                "api": {
                    "port": 7777,
                    "workers": 2
                },
                "logging": {
                    "level": "DEBUG"
                }
            }
        }
        yaml.dump(custom_config, f)
        config_path = f.name
    
    try:
        print(f"\nLoading custom config from: {config_path}")
        config = Config(config_path=Path(config_path), env="development")
        
        print(f"  API Port: {config.api.port}")
        print(f"  API Workers: {config.api.workers}")
        print(f"  Log Level: {config.logging.level}")
    finally:
        # Clean up
        os.unlink(config_path)


def demo_validation():
    """Demonstrate configuration validation"""
    print("\n" + "=" * 60)
    print("Configuration Validation")
    print("=" * 60)
    
    from ara.core.exceptions import ConfigurationError
    
    print("\nTesting invalid configurations...")
    
    # Test invalid port
    try:
        config = Config(env="development")
        config.api.port = 99999
        config._validate()
        print("  ❌ Port validation failed to catch error")
    except ConfigurationError as e:
        print(f"  ✓ Invalid port caught: {str(e)[:60]}...")
    
    # Test invalid log level
    try:
        config = Config(env="development")
        config.logging.level = "INVALID"
        config._validate()
        print("  ❌ Log level validation failed to catch error")
    except ConfigurationError as e:
        print(f"  ✓ Invalid log level caught: {str(e)[:60]}...")
    
    # Test invalid learning rate
    try:
        config = Config(env="development")
        config.model.learning_rate = -0.1
        config._validate()
        print("  ❌ Learning rate validation failed to catch error")
    except ConfigurationError as e:
        print(f"  ✓ Invalid learning rate caught: {str(e)[:60]}...")
    
    print("\n  All validations working correctly! ✓")


def demo_save_load():
    """Demonstrate saving and loading configuration"""
    print("\n" + "=" * 60)
    print("Save and Load Configuration")
    print("=" * 60)
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "my_config.yaml"
        
        # Create and modify config
        print("\nCreating configuration...")
        config1 = Config(config_path=config_path, env="development")
        config1.api.port = 8888
        config1.model.batch_size = 64
        config1.logging.level = "WARNING"
        
        print(f"  API Port: {config1.api.port}")
        print(f"  Batch Size: {config1.model.batch_size}")
        print(f"  Log Level: {config1.logging.level}")
        
        # Save config
        print(f"\nSaving to: {config_path}")
        config1.save()
        
        # Load config
        print("\nLoading configuration...")
        config2 = Config(config_path=config_path, env="development")
        
        print(f"  API Port: {config2.api.port}")
        print(f"  Batch Size: {config2.model.batch_size}")
        print(f"  Log Level: {config2.logging.level}")
        
        print("\n  Configuration saved and loaded successfully! ✓")


def demo_create_default():
    """Demonstrate creating default configuration"""
    print("\n" + "=" * 60)
    print("Create Default Configuration")
    print("=" * 60)
    
    import tempfile
    import yaml
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "default_config.yaml"
        
        print(f"\nCreating default config at: {config_path}")
        Config.create_default_config(config_path)
        
        # Load and display
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        print("\nEnvironments created:")
        for env in config_data.keys():
            print(f"  - {env}")
        
        print("\n  Default configuration created successfully! ✓")


def main():
    """Run all demos"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "ARA AI Configuration System Demo" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        demo_basic_usage()
        demo_environment_switching()
        demo_env_overrides()
        demo_custom_config()
        demo_validation()
        demo_save_load()
        demo_create_default()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully! ✓")
        print("=" * 60)
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
