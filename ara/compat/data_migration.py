"""
Data Migration Tools.

This module provides tools for migrating data between different versions:
- Cache format migration
- Model architecture migration
- Configuration migration
- Database schema migration
- Data validation after migration
"""

import json
import pickle
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import numpy as np

from ara.core.exceptions import ModelError, ValidationError

logger = logging.getLogger(__name__)


class CacheMigrator:
    """Migrate cache data from old format to new format."""
    
    def __init__(self, old_cache_dir: Path, new_cache_dir: Path):
        """
        Initialize cache migrator.
        
        Args:
            old_cache_dir: Directory with old cache format
            new_cache_dir: Directory for new cache format
        """
        self.old_cache_dir = Path(old_cache_dir)
        self.new_cache_dir = Path(new_cache_dir)
        self.new_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.migration_log = []
    
    def migrate(self) -> Dict[str, Any]:
        """
        Migrate all cache files.
        
        Returns:
            Migration statistics
        """
        logger.info(f"Starting cache migration from {self.old_cache_dir} to {self.new_cache_dir}")
        
        stats = {
            'total_files': 0,
            'migrated': 0,
            'skipped': 0,
            'failed': 0,
            'errors': []
        }
        
        if not self.old_cache_dir.exists():
            logger.warning(f"Old cache directory not found: {self.old_cache_dir}")
            return stats
        
        # Find all cache files
        cache_files = list(self.old_cache_dir.glob("**/*.pkl")) + \
                     list(self.old_cache_dir.glob("**/*.json"))
        
        stats['total_files'] = len(cache_files)
        
        for cache_file in cache_files:
            try:
                self._migrate_file(cache_file)
                stats['migrated'] += 1
            except Exception as e:
                logger.error(f"Failed to migrate {cache_file}: {e}")
                stats['failed'] += 1
                stats['errors'].append({
                    'file': str(cache_file),
                    'error': str(e)
                })
        
        # Save migration log
        self._save_migration_log()
        
        logger.info(f"Cache migration complete: {stats}")
        return stats
    
    def _migrate_file(self, old_file: Path) -> None:
        """Migrate a single cache file."""
        # Determine relative path
        rel_path = old_file.relative_to(self.old_cache_dir)
        new_file = self.new_cache_dir / rel_path
        new_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load old format
        if old_file.suffix == '.pkl':
            with open(old_file, 'rb') as f:
                data = pickle.load(f)
        elif old_file.suffix == '.json':
            with open(old_file, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {old_file.suffix}")
        
        # Convert to new format
        new_data = self._convert_cache_format(data)
        
        # Save new format
        if new_file.suffix == '.pkl':
            with open(new_file, 'wb') as f:
                pickle.dump(new_data, f)
        elif new_file.suffix == '.json':
            with open(new_file, 'w') as f:
                json.dump(new_data, f, indent=2)
        
        self.migration_log.append({
            'timestamp': datetime.now().isoformat(),
            'old_file': str(old_file),
            'new_file': str(new_file),
            'status': 'success'
        })
    
    def _convert_cache_format(self, old_data: Any) -> Any:
        """
        Convert old cache format to new format.
        
        Old format: Simple dict with data
        New format: Dict with metadata and versioning
        """
        if isinstance(old_data, dict) and 'version' in old_data:
            # Already in new format
            return old_data
        
        # Wrap in new format
        new_data = {
            'version': '2.0',
            'created_at': datetime.now().isoformat(),
            'data': old_data,
            'metadata': {
                'migrated': True,
                'migration_date': datetime.now().isoformat()
            }
        }
        
        return new_data
    
    def _save_migration_log(self) -> None:
        """Save migration log."""
        log_file = self.new_cache_dir / 'migration_log.json'
        with open(log_file, 'w') as f:
            json.dump(self.migration_log, f, indent=2)


class ModelMigrator:
    """Migrate models from old architecture to new architecture."""
    
    def __init__(self, old_models_dir: Path, new_models_dir: Path):
        """
        Initialize model migrator.
        
        Args:
            old_models_dir: Directory with old models
            new_models_dir: Directory for new models
        """
        self.old_models_dir = Path(old_models_dir)
        self.new_models_dir = Path(new_models_dir)
        self.new_models_dir.mkdir(parents=True, exist_ok=True)
        
        self.migration_log = []
    
    def migrate(self) -> Dict[str, Any]:
        """
        Migrate all models.
        
        Returns:
            Migration statistics
        """
        logger.info(f"Starting model migration from {self.old_models_dir} to {self.new_models_dir}")
        
        stats = {
            'total_models': 0,
            'migrated': 0,
            'skipped': 0,
            'failed': 0,
            'errors': []
        }
        
        if not self.old_models_dir.exists():
            logger.warning(f"Old models directory not found: {self.old_models_dir}")
            return stats
        
        # Find all model files
        model_files = list(self.old_models_dir.glob("*.pkl"))
        stats['total_models'] = len(model_files)
        
        for model_file in model_files:
            try:
                self._migrate_model(model_file)
                stats['migrated'] += 1
            except Exception as e:
                logger.error(f"Failed to migrate {model_file}: {e}")
                stats['failed'] += 1
                stats['errors'].append({
                    'file': str(model_file),
                    'error': str(e)
                })
        
        # Save migration log
        self._save_migration_log()
        
        logger.info(f"Model migration complete: {stats}")
        return stats
    
    def _migrate_model(self, old_model_file: Path) -> None:
        """Migrate a single model."""
        # Load old model
        with open(old_model_file, 'rb') as f:
            old_model = pickle.load(f)
        
        # Convert to new architecture
        new_model = self._convert_model_architecture(old_model)
        
        # Save new model
        new_model_file = self.new_models_dir / old_model_file.name
        with open(new_model_file, 'wb') as f:
            pickle.dump(new_model, f)
        
        # Create metadata file
        metadata = self._create_model_metadata(old_model_file, new_model)
        metadata_file = new_model_file.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.migration_log.append({
            'timestamp': datetime.now().isoformat(),
            'old_file': str(old_model_file),
            'new_file': str(new_model_file),
            'status': 'success'
        })
    
    def _convert_model_architecture(self, old_model: Any) -> Any:
        """
        Convert old model architecture to new architecture.
        
        This is a placeholder - actual conversion depends on specific changes.
        """
        # Check if model has new architecture attributes
        if hasattr(old_model, 'model_version') and old_model.model_version >= 2.0:
            return old_model
        
        # Add new architecture attributes
        if hasattr(old_model, '__dict__'):
            old_model.model_version = 2.0
            old_model.migration_date = datetime.now().isoformat()
            old_model.migrated = True
        
        return old_model
    
    def _create_model_metadata(self, old_file: Path, model: Any) -> Dict[str, Any]:
        """Create metadata for migrated model."""
        return {
            'model_name': old_file.stem,
            'version': '2.0',
            'migrated': True,
            'migration_date': datetime.now().isoformat(),
            'original_file': str(old_file),
            'model_type': type(model).__name__,
            'file_size': old_file.stat().st_size,
            'file_hash': self._calculate_file_hash(old_file)
        }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _save_migration_log(self) -> None:
        """Save migration log."""
        log_file = self.new_models_dir / 'migration_log.json'
        with open(log_file, 'w') as f:
            json.dump(self.migration_log, f, indent=2)


class ConfigMigrator:
    """Migrate configuration files."""
    
    def __init__(self, old_config_file: Path, new_config_file: Path):
        """
        Initialize config migrator.
        
        Args:
            old_config_file: Old configuration file
            new_config_file: New configuration file
        """
        self.old_config_file = Path(old_config_file)
        self.new_config_file = Path(new_config_file)
        self.new_config_file.parent.mkdir(parents=True, exist_ok=True)
    
    def migrate(self) -> Dict[str, Any]:
        """
        Migrate configuration.
        
        Returns:
            Migration statistics
        """
        logger.info(f"Migrating config from {self.old_config_file} to {self.new_config_file}")
        
        if not self.old_config_file.exists():
            logger.warning(f"Old config file not found: {self.old_config_file}")
            return {'status': 'skipped', 'reason': 'file_not_found'}
        
        # Load old config
        if self.old_config_file.suffix in ['.yaml', '.yml']:
            import yaml
            with open(self.old_config_file, 'r') as f:
                old_config = yaml.safe_load(f)
        elif self.old_config_file.suffix == '.json':
            with open(self.old_config_file, 'r') as f:
                old_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {self.old_config_file.suffix}")
        
        # Convert to new format
        new_config = self._convert_config_format(old_config)
        
        # Save new config
        if self.new_config_file.suffix in ['.yaml', '.yml']:
            import yaml
            with open(self.new_config_file, 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)
        elif self.new_config_file.suffix == '.json':
            with open(self.new_config_file, 'w') as f:
                json.dump(new_config, f, indent=2)
        
        logger.info(f"Config migration complete")
        return {
            'status': 'success',
            'old_file': str(self.old_config_file),
            'new_file': str(self.new_config_file)
        }
    
    def _convert_config_format(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert old config format to new format.
        
        Maps old keys to new keys and adds new default values.
        """
        # Check if already in new format
        if 'version' in old_config and old_config['version'] >= 2.0:
            return old_config
        
        # Create new config structure
        new_config = {
            'version': 2.0,
            'migrated': True,
            'migration_date': datetime.now().isoformat()
        }
        
        # Map old keys to new keys
        key_mapping = {
            'model_path': 'models.path',
            'cache_dir': 'cache.directory',
            'api_key': 'api.key',
            'data_provider': 'data.provider'
        }
        
        for old_key, new_key in key_mapping.items():
            if old_key in old_config:
                # Handle nested keys
                keys = new_key.split('.')
                current = new_config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = old_config[old_key]
        
        # Add new default values
        defaults = {
            'models': {
                'registry_dir': 'models/registry',
                'validation_dir': 'models/validation'
            },
            'cache': {
                'ttl': 3600,
                'max_size_mb': 1024
            },
            'api': {
                'rate_limit': 100,
                'timeout': 30
            }
        }
        
        # Merge defaults (don't overwrite existing values)
        self._merge_dicts(new_config, defaults)
        
        return new_config
    
    def _merge_dicts(self, target: Dict, source: Dict) -> None:
        """Merge source dict into target dict (non-destructive)."""
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, dict) and isinstance(target[key], dict):
                self._merge_dicts(target[key], value)


class DatabaseMigrator:
    """Migrate database schema."""
    
    def __init__(self, db_path: Path):
        """
        Initialize database migrator.
        
        Args:
            db_path: Path to database file
        """
        self.db_path = Path(db_path)
        self.migrations = []
    
    def add_migration(self, version: str, migration_func: callable) -> None:
        """
        Add a migration.
        
        Args:
            version: Migration version
            migration_func: Function to execute migration
        """
        self.migrations.append({
            'version': version,
            'func': migration_func
        })
    
    def migrate(self) -> Dict[str, Any]:
        """
        Execute all pending migrations.
        
        Returns:
            Migration statistics
        """
        logger.info(f"Starting database migration for {self.db_path}")
        
        stats = {
            'total_migrations': len(self.migrations),
            'executed': 0,
            'skipped': 0,
            'failed': 0,
            'errors': []
        }
        
        # Get current version
        current_version = self._get_current_version()
        
        # Execute migrations
        for migration in sorted(self.migrations, key=lambda m: m['version']):
            if migration['version'] <= current_version:
                stats['skipped'] += 1
                continue
            
            try:
                logger.info(f"Executing migration {migration['version']}")
                migration['func'](self.db_path)
                self._update_version(migration['version'])
                stats['executed'] += 1
            except Exception as e:
                logger.error(f"Migration {migration['version']} failed: {e}")
                stats['failed'] += 1
                stats['errors'].append({
                    'version': migration['version'],
                    'error': str(e)
                })
                break  # Stop on first failure
        
        logger.info(f"Database migration complete: {stats}")
        return stats
    
    def _get_current_version(self) -> str:
        """Get current database version."""
        version_file = self.db_path.parent / 'db_version.txt'
        if version_file.exists():
            return version_file.read_text().strip()
        return '0.0'
    
    def _update_version(self, version: str) -> None:
        """Update database version."""
        version_file = self.db_path.parent / 'db_version.txt'
        version_file.write_text(version)


class DataValidator:
    """Validate data after migration."""
    
    def __init__(self):
        """Initialize data validator."""
        self.validation_results = []
    
    def validate_cache(self, cache_dir: Path) -> Dict[str, Any]:
        """
        Validate cache data.
        
        Args:
            cache_dir: Cache directory to validate
            
        Returns:
            Validation results
        """
        logger.info(f"Validating cache in {cache_dir}")
        
        results = {
            'total_files': 0,
            'valid': 0,
            'invalid': 0,
            'errors': []
        }
        
        cache_files = list(cache_dir.glob("**/*.pkl")) + list(cache_dir.glob("**/*.json"))
        results['total_files'] = len(cache_files)
        
        for cache_file in cache_files:
            try:
                # Try to load file
                if cache_file.suffix == '.pkl':
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                elif cache_file.suffix == '.json':
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                
                # Validate structure
                if isinstance(data, dict) and 'version' in data:
                    results['valid'] += 1
                else:
                    results['invalid'] += 1
                    results['errors'].append({
                        'file': str(cache_file),
                        'error': 'Missing version field'
                    })
            except Exception as e:
                results['invalid'] += 1
                results['errors'].append({
                    'file': str(cache_file),
                    'error': str(e)
                })
        
        logger.info(f"Cache validation complete: {results}")
        return results
    
    def validate_models(self, models_dir: Path) -> Dict[str, Any]:
        """
        Validate model files.
        
        Args:
            models_dir: Models directory to validate
            
        Returns:
            Validation results
        """
        logger.info(f"Validating models in {models_dir}")
        
        results = {
            'total_models': 0,
            'valid': 0,
            'invalid': 0,
            'errors': []
        }
        
        model_files = list(models_dir.glob("*.pkl"))
        results['total_models'] = len(model_files)
        
        for model_file in model_files:
            try:
                # Try to load model
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                # Check for metadata file
                metadata_file = model_file.with_suffix('.json')
                if not metadata_file.exists():
                    results['invalid'] += 1
                    results['errors'].append({
                        'file': str(model_file),
                        'error': 'Missing metadata file'
                    })
                    continue
                
                # Validate metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                required_fields = ['model_name', 'version', 'model_type']
                missing_fields = [f for f in required_fields if f not in metadata]
                
                if missing_fields:
                    results['invalid'] += 1
                    results['errors'].append({
                        'file': str(model_file),
                        'error': f'Missing metadata fields: {missing_fields}'
                    })
                else:
                    results['valid'] += 1
                    
            except Exception as e:
                results['invalid'] += 1
                results['errors'].append({
                    'file': str(model_file),
                    'error': str(e)
                })
        
        logger.info(f"Model validation complete: {results}")
        return results
    
    def validate_config(self, config_file: Path) -> Dict[str, Any]:
        """
        Validate configuration file.
        
        Args:
            config_file: Configuration file to validate
            
        Returns:
            Validation results
        """
        logger.info(f"Validating config {config_file}")
        
        results = {
            'valid': False,
            'errors': []
        }
        
        try:
            # Load config
            if config_file.suffix in ['.yaml', '.yml']:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
            elif config_file.suffix == '.json':
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                results['errors'].append(f"Unsupported config format: {config_file.suffix}")
                return results
            
            # Validate structure
            if not isinstance(config, dict):
                results['errors'].append("Config must be a dictionary")
                return results
            
            if 'version' not in config:
                results['errors'].append("Missing version field")
            
            if not results['errors']:
                results['valid'] = True
                
        except Exception as e:
            results['errors'].append(str(e))
        
        logger.info(f"Config validation complete: {results}")
        return results


def migrate_all(
    old_root: Path,
    new_root: Path,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Migrate all data from old structure to new structure.
    
    Args:
        old_root: Root directory of old data
        new_root: Root directory for new data
        validate: Whether to validate after migration
        
    Returns:
        Combined migration statistics
    """
    old_root = Path(old_root)
    new_root = Path(new_root)
    new_root.mkdir(parents=True, exist_ok=True)
    
    results = {
        'cache': {},
        'models': {},
        'config': {},
        'validation': {}
    }
    
    # Migrate cache
    if (old_root / 'cache').exists():
        cache_migrator = CacheMigrator(
            old_root / 'cache',
            new_root / 'cache'
        )
        results['cache'] = cache_migrator.migrate()
    
    # Migrate models
    if (old_root / 'models').exists():
        model_migrator = ModelMigrator(
            old_root / 'models',
            new_root / 'models'
        )
        results['models'] = model_migrator.migrate()
    
    # Migrate config
    old_config = old_root / 'config.yaml'
    if old_config.exists():
        config_migrator = ConfigMigrator(
            old_config,
            new_root / 'config.yaml'
        )
        results['config'] = config_migrator.migrate()
    
    # Validate if requested
    if validate:
        validator = DataValidator()
        
        if (new_root / 'cache').exists():
            results['validation']['cache'] = validator.validate_cache(new_root / 'cache')
        
        if (new_root / 'models').exists():
            results['validation']['models'] = validator.validate_models(new_root / 'models')
        
        if (new_root / 'config.yaml').exists():
            results['validation']['config'] = validator.validate_config(new_root / 'config.yaml')
    
    logger.info(f"Complete migration finished: {results}")
    return results
