# ARA AI Configuration

This directory contains the configuration management system for ARA AI.

## Configuration Files

- **config.yaml** - Active configuration file (used by the system)
- **config.example.yaml** - Complete configuration template with all options documented
- **config.minimal.yaml** - Minimal configuration template with only essential settings

## Environments

ARA AI supports three environments:

1. **development** - Local development with verbose logging and relaxed settings
2. **staging** - Pre-production testing environment with production-like settings
3. **production** - Production environment with optimized performance and security

## Quick Start

### 1. Create Configuration File

Copy the example configuration:

```bash
cp ara/config/config.example.yaml ara/config/config.yaml
```

Or create a minimal configuration:

```bash
cp ara/config/config.minimal.yaml ara/config/config.yaml
```

### 2. Set Environment

Set the environment using the `ARA_ENV` environment variable:

```bash
# Development (default)
export ARA_ENV=development

# Staging
export ARA_ENV=staging

# Production
export ARA_ENV=production
```

### 3. Configure Settings

Edit `config.yaml` to customize settings for your environment.

## Configuration Structure

```yaml
<environment>:
  data:
    cache_ttl: 300          # Cache TTL in seconds
    max_retries: 3          # API retry attempts
    timeout: 30             # Request timeout
    providers:              # Data provider configurations
      yfinance:
        enabled: true
        priority: 1
  
  model:
    model_dir: "models"     # Model storage directory
    default_ensemble_size: 12
    gpu_enabled: true
    batch_size: 32
    epochs: 200
    learning_rate: 0.0005
  
  api:
    host: "0.0.0.0"
    port: 8000
    workers: 4
    rate_limit: 100         # Requests per minute
    enable_cors: true
  
  cache:
    enabled: true
    redis_url: null         # Redis URL or null for in-memory
    ttl_l1: 10             # L1 cache TTL (seconds)
    ttl_l2: 300            # L2 cache TTL (seconds)
    max_size_mb: 1000
  
  logging:
    level: "INFO"           # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format: "json"          # json, text, structured
    file: null              # Log file path or null
    console: true
```

## Environment Variables

Configuration values can be overridden using environment variables:

### General
- `ARA_ENV` - Environment name (development, staging, production)

### Data Configuration
- `ARA_CACHE_TTL` - Cache TTL in seconds
- `ARA_BINANCE_API_KEY` - Binance API key
- `ARA_BINANCE_API_SECRET` - Binance API secret
- `ARA_COINBASE_API_KEY` - Coinbase API key
- `ARA_ALPHA_VANTAGE_API_KEY` - Alpha Vantage API key
- `ARA_POLYGON_API_KEY` - Polygon.io API key

### Model Configuration
- `ARA_MODEL_DIR` - Model directory path
- `ARA_GPU_ENABLED` - Enable GPU (true/false)

### API Configuration
- `ARA_API_HOST` - API host
- `ARA_API_PORT` - API port

### Cache Configuration
- `ARA_REDIS_URL` - Redis connection URL (e.g., redis://localhost:6379/0)

### Logging Configuration
- `ARA_LOG_LEVEL` - Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Configuration Priority

Configuration values are loaded in the following priority (highest to lowest):

1. **Environment Variables** - `ARA_*` variables override everything
2. **YAML Configuration** - Values from config.yaml
3. **Default Values** - Built-in defaults

## Usage in Code

### Load Configuration

```python
from ara.config import get_config

# Get global configuration instance
config = get_config()

# Access configuration values
print(config.env)  # Current environment
print(config.data.cache_ttl)
print(config.model.gpu_enabled)
print(config.api.port)
```

### Reload Configuration

```python
from ara.config import get_config

# Force reload configuration
config = get_config(reload=True)
```

### Create Custom Configuration

```python
from ara.config import Config
from pathlib import Path

# Load from custom path
config = Config(
    config_path=Path("/custom/path/config.yaml"),
    env="production"
)

# Check environment
if config.is_production():
    print("Running in production mode")
```

### Save Configuration

```python
from ara.config import get_config

config = get_config()

# Modify configuration
config.api.port = 9000
config.logging.level = "DEBUG"

# Save to file
config.save()
```

### Create Default Configuration

```python
from ara.config import Config
from pathlib import Path

# Create default config file with all environments
Config.create_default_config(
    path=Path("config.yaml"),
    environments=["development", "staging", "production"]
)
```

## Validation

The configuration system validates all values:

- **Type checking** - Ensures correct data types
- **Range validation** - Checks values are within acceptable ranges
- **Format validation** - Validates URLs, paths, and other formats
- **Cross-validation** - Checks relationships between settings

Invalid configurations will raise a `ConfigurationError` with details about the issue.

## Environment-Specific Defaults

### Development
- Verbose logging (INFO level)
- Console output enabled
- CORS enabled
- In-memory cache (no Redis required)
- Smaller batch sizes and fewer epochs

### Staging
- Moderate logging (INFO level)
- File and console logging
- CORS enabled
- Redis cache recommended
- Production-like settings for testing

### Production
- Minimal logging (WARNING level)
- File logging only
- CORS disabled
- Redis cache required
- Optimized for performance and security
- More workers and larger batch sizes

## Best Practices

1. **Never commit secrets** - Use environment variables for API keys
2. **Use appropriate environment** - Don't run production config in development
3. **Validate after changes** - Configuration is validated on load
4. **Use Redis in production** - In-memory cache is not suitable for production
5. **Monitor log files** - Ensure log files don't grow too large
6. **Backup configurations** - Keep backups of production configs
7. **Document custom settings** - Add comments for non-standard values

## Troubleshooting

### Configuration Not Loading

Check that:
1. Config file exists at expected location
2. YAML syntax is valid
3. Environment name is correct
4. File permissions allow reading

### Validation Errors

Read the error message carefully - it will indicate:
- Which configuration value is invalid
- What the valid range or format is
- The current invalid value

### Environment Variables Not Working

Ensure:
1. Variables are prefixed with `ARA_`
2. Variable names match exactly (case-sensitive)
3. Values are in correct format (e.g., "true"/"false" for booleans)
4. Environment is properly set before running

## Examples

### Development Setup

```bash
# Use default development config
export ARA_ENV=development
python ara.py predict AAPL
```

### Production Setup

```bash
# Set production environment
export ARA_ENV=production

# Override specific settings
export ARA_API_PORT=9000
export ARA_REDIS_URL=redis://prod-redis:6379/0
export ARA_LOG_LEVEL=ERROR

# Run application
python ara.py predict AAPL
```

### Custom Configuration

```bash
# Use custom config file
export ARA_CONFIG_PATH=/etc/ara/custom-config.yaml
export ARA_ENV=production

python ara.py predict AAPL
```

## Support

For issues or questions about configuration:
1. Check this README
2. Review config.example.yaml for all options
3. Check the troubleshooting guide
4. Open an issue on GitHub
