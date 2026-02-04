"""
Model and Data Migration Utilities

Provides tools for migrating old model formats and data structures
to the new ARA architecture.
"""

import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import warnings


class ModelMigrator:
    """
    Migrates old model formats to new format

    Handles migration of:
    - Old pickle models to new format
    - Model metadata and versioning
    - Configuration files
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize model migrator

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.migration_log: List[Dict[str, Any]] = []

    def migrate_model_directory(
        self, old_dir: Path, new_dir: Path, backup: bool = True
    ) -> Dict[str, Any]:
        """
        Migrate entire model directory from old to new format

        Args:
            old_dir: Path to old models directory
            new_dir: Path to new models directory
            backup: Whether to create backup of old models

        Returns:
            Migration summary
        """
        old_dir = Path(old_dir)
        new_dir = Path(new_dir)

        if not old_dir.exists():
            raise FileNotFoundError(f"Old model directory not found: {old_dir}")

        # Create new directory
        new_dir.mkdir(parents=True, exist_ok=True)

        # Create backup if requested
        if backup:
            backup_dir = (
                old_dir.parent / f"{old_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            if self.verbose:
                print(f"Creating backup: {backup_dir}")
            shutil.copytree(old_dir, backup_dir)

        # Find all model files
        model_files = list(old_dir.glob("*.pkl")) + list(old_dir.glob("*.pt"))

        migrated_count = 0
        failed_count = 0

        for model_file in model_files:
            try:
                if self.verbose:
                    print(f"Migrating: {model_file.name}")

                result = self.migrate_model_file(model_file, new_dir)

                if result["success"]:
                    migrated_count += 1
                else:
                    failed_count += 1

                self.migration_log.append(result)

            except Exception as e:
                if self.verbose:
                    print(f"Failed to migrate {model_file.name}: {e}")

                failed_count += 1
                self.migration_log.append(
                    {
                        "file": str(model_file),
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        summary = {
            "total_files": len(model_files),
            "migrated": migrated_count,
            "failed": failed_count,
            "old_dir": str(old_dir),
            "new_dir": str(new_dir),
            "backup_created": backup,
            "timestamp": datetime.now().isoformat(),
        }

        # Save migration log
        log_file = new_dir / "migration_log.json"
        with open(log_file, "w") as f:
            json.dump({"summary": summary, "details": self.migration_log}, f, indent=2)

        if self.verbose:
            print("\nMigration Summary:")
            print(f"  Total files: {summary['total_files']}")
            print(f"  Migrated: {summary['migrated']}")
            print(f"  Failed: {summary['failed']}")
            print(f"  Log saved to: {log_file}")

        return summary

    def migrate_model_file(self, old_file: Path, new_dir: Path) -> Dict[str, Any]:
        """
        Migrate a single model file

        Args:
            old_file: Path to old model file
            new_dir: Path to new models directory

        Returns:
            Migration result
        """
        old_file = Path(old_file)
        new_dir = Path(new_dir)

        try:
            # Determine file type
            if old_file.suffix == ".pkl":
                return self._migrate_pickle_model(old_file, new_dir)
            elif old_file.suffix == ".pt":
                return self._migrate_pytorch_model(old_file, new_dir)
            elif old_file.suffix == ".json":
                return self._migrate_json_model(old_file, new_dir)
            else:
                return {
                    "file": str(old_file),
                    "success": False,
                    "error": f"Unsupported file type: {old_file.suffix}",
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            return {
                "file": str(old_file),
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _migrate_pickle_model(self, old_file: Path, new_dir: Path) -> Dict[str, Any]:
        """Migrate pickle model file"""
        try:
            # Load old model
            with open(old_file, "rb") as f:
                old_model = pickle.load(f)

            # Create new model metadata
            metadata = {
                "model_name": old_file.stem,
                "original_file": str(old_file),
                "migrated_from": "v3.x",
                "migrated_to": "v4.0",
                "migration_date": datetime.now().isoformat(),
                "model_type": (
                    type(old_model).__name__ if hasattr(old_model, "__name__") else "unknown"
                ),
            }

            # Save model in new format (still pickle, but with metadata)
            new_file = new_dir / old_file.name
            with open(new_file, "wb") as f:
                pickle.dump(old_model, f)

            # Save metadata
            metadata_file = new_dir / f"{old_file.stem}_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            return {
                "file": str(old_file),
                "new_file": str(new_file),
                "metadata_file": str(metadata_file),
                "success": True,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            raise Exception(f"Failed to migrate pickle model: {e}")

    def _migrate_pytorch_model(self, old_file: Path, new_dir: Path) -> Dict[str, Any]:
        """Migrate PyTorch model file"""
        try:
            # Simply copy PyTorch models (they're already in good format)
            new_file = new_dir / old_file.name
            shutil.copy2(old_file, new_file)

            # Create metadata
            metadata = {
                "model_name": old_file.stem,
                "original_file": str(old_file),
                "migrated_from": "v3.x",
                "migrated_to": "v4.0",
                "migration_date": datetime.now().isoformat(),
                "model_type": "pytorch",
            }

            metadata_file = new_dir / f"{old_file.stem}_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            return {
                "file": str(old_file),
                "new_file": str(new_file),
                "metadata_file": str(metadata_file),
                "success": True,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            raise Exception(f"Failed to migrate PyTorch model: {e}")

    def _migrate_json_model(self, old_file: Path, new_dir: Path) -> Dict[str, Any]:
        """Migrate JSON model configuration"""
        try:
            # Load old config
            with open(old_file, "r") as f:
                old_config = json.load(f)

            # Update config format if needed
            new_config = self._update_config_format(old_config)

            # Save new config
            new_file = new_dir / old_file.name
            with open(new_file, "w") as f:
                json.dump(new_config, f, indent=2)

            return {
                "file": str(old_file),
                "new_file": str(new_file),
                "success": True,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            raise Exception(f"Failed to migrate JSON model: {e}")

    def _update_config_format(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration format from old to new"""
        new_config = old_config.copy()

        # Add version info
        new_config["version"] = "4.0.0"
        new_config["migrated_from"] = old_config.get("version", "3.x")
        new_config["migration_date"] = datetime.now().isoformat()

        # Update any deprecated keys
        key_mappings = {
            "model_path": "model_file",
            "cache_dir": "cache_directory",
            # Add more mappings as needed
        }

        for old_key, new_key in key_mappings.items():
            if old_key in new_config:
                new_config[new_key] = new_config.pop(old_key)

        return new_config

    def get_migration_log(self) -> List[Dict[str, Any]]:
        """Get migration log"""
        return self.migration_log


class DataMigrator:
    """
    Migrates old data formats and cache structures

    Handles migration of:
    - Cache files
    - Prediction history
    - Configuration files
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize data migrator

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.migration_log: List[Dict[str, Any]] = []

    def migrate_cache_directory(
        self, old_cache_dir: Path, new_cache_dir: Path, backup: bool = True
    ) -> Dict[str, Any]:
        """
        Migrate cache directory from old to new format

        Args:
            old_cache_dir: Path to old cache directory
            new_cache_dir: Path to new cache directory
            backup: Whether to create backup

        Returns:
            Migration summary
        """
        old_cache_dir = Path(old_cache_dir)
        new_cache_dir = Path(new_cache_dir)

        if not old_cache_dir.exists():
            warnings.warn(f"Old cache directory not found: {old_cache_dir}")
            return {"success": False, "error": "Old cache directory not found"}

        # Create new directory
        new_cache_dir.mkdir(parents=True, exist_ok=True)

        # Create backup if requested
        if backup:
            backup_dir = (
                old_cache_dir.parent
                / f"{old_cache_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            if self.verbose:
                print(f"Creating backup: {backup_dir}")
            shutil.copytree(old_cache_dir, backup_dir)

        # Migrate cache files
        cache_files = list(old_cache_dir.glob("*.json"))

        migrated_count = 0
        failed_count = 0

        for cache_file in cache_files:
            try:
                if self.verbose:
                    print(f"Migrating cache: {cache_file.name}")

                result = self._migrate_cache_file(cache_file, new_cache_dir)

                if result["success"]:
                    migrated_count += 1
                else:
                    failed_count += 1

                self.migration_log.append(result)

            except Exception as e:
                if self.verbose:
                    print(f"Failed to migrate {cache_file.name}: {e}")

                failed_count += 1
                self.migration_log.append(
                    {
                        "file": str(cache_file),
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        summary = {
            "total_files": len(cache_files),
            "migrated": migrated_count,
            "failed": failed_count,
            "old_dir": str(old_cache_dir),
            "new_dir": str(new_cache_dir),
            "backup_created": backup,
            "timestamp": datetime.now().isoformat(),
        }

        if self.verbose:
            print("\nCache Migration Summary:")
            print(f"  Total files: {summary['total_files']}")
            print(f"  Migrated: {summary['migrated']}")
            print(f"  Failed: {summary['failed']}")

        return summary

    def _migrate_cache_file(self, old_file: Path, new_dir: Path) -> Dict[str, Any]:
        """Migrate a single cache file"""
        try:
            # Load old cache data
            with open(old_file, "r") as f:
                old_data = json.load(f)

            # Update format
            new_data = self._update_cache_format(old_data)

            # Save to new location
            new_file = new_dir / old_file.name
            with open(new_file, "w") as f:
                json.dump(new_data, f, indent=2)

            return {
                "file": str(old_file),
                "new_file": str(new_file),
                "success": True,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "file": str(old_file),
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _update_cache_format(self, old_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update cache data format"""
        new_data = old_data.copy()

        # Add version info
        new_data["cache_version"] = "4.0.0"
        new_data["migrated_from"] = old_data.get("version", "3.x")
        new_data["migration_date"] = datetime.now().isoformat()

        # Update any deprecated fields
        if "predictions" in new_data:
            for pred in new_data["predictions"]:
                # Ensure all required fields exist
                if "confidence" not in pred:
                    pred["confidence"] = 0.75  # Default confidence

                # Update date format if needed
                if "date" in pred and not isinstance(pred["date"], str):
                    pred["date"] = str(pred["date"])

        return new_data

    def validate_migration(self, new_dir: Path) -> Dict[str, Any]:
        """
        Validate migrated data

        Args:
            new_dir: Path to new data directory

        Returns:
            Validation results
        """
        new_dir = Path(new_dir)

        if not new_dir.exists():
            return {"valid": False, "error": "New directory does not exist"}

        # Check for required files
        files = list(new_dir.glob("*"))

        validation_results = {
            "valid": True,
            "total_files": len(files),
            "issues": [],
            "timestamp": datetime.now().isoformat(),
        }

        # Validate each file
        for file in files:
            if file.suffix == ".json":
                try:
                    with open(file, "r") as f:
                        data = json.load(f)

                    # Check for required fields
                    if "cache_version" not in data and "version" not in data:
                        validation_results["issues"].append(
                            {"file": str(file), "issue": "Missing version information"}
                        )
                        validation_results["valid"] = False

                except Exception as e:
                    validation_results["issues"].append(
                        {"file": str(file), "issue": f"Failed to load: {e}"}
                    )
                    validation_results["valid"] = False

        if self.verbose:
            print("\nValidation Results:")
            print(f"  Valid: {validation_results['valid']}")
            print(f"  Total files: {validation_results['total_files']}")
            print(f"  Issues: {len(validation_results['issues'])}")

        return validation_results
