"""
Backward Compatibility Layer Demo

This example demonstrates how to use the backward compatibility layer
to run old code with the new ARA AI architecture.

WARNING: This compatibility layer is deprecated and will be removed in v5.0.0
Please migrate to the new API as soon as possible.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings

# Example 1: Using old API with compatibility layer
print("=" * 70)
print("Example 1: Using Old API (Deprecated)")
print("=" * 70)

# This will issue deprecation warnings
from ara.compat import predict_stock

print("\nAttempting prediction with old API...")
print("Note: You will see deprecation warnings\n")

# This works but is deprecated
result = predict_stock('AAPL', days=5, verbose=False)

if result and 'error' not in result:
    print(f"[OK] Prediction successful (using compatibility layer)")
    print(f"  Symbol: {result.get('symbol', 'N/A')}")
    print(f"  Current Price: ${result.get('current_price', 0):.2f}")
else:
    print(f"[FAIL] Prediction failed or engine not available")
    if result:
        print(f"  Error: {result.get('error', 'Unknown')}")

# Example 2: Using old AraAI class
print("\n" + "=" * 70)
print("Example 2: Using Old AraAI Class (Deprecated)")
print("=" * 70)

from ara.compat import AraAI

print("\nInitializing AraAI with compatibility layer...")
ara = AraAI(verbose=False)

# Get system info
info = ara.get_system_info()
print(f"\nSystem Info:")
print(f"  Version: {info.get('version', 'N/A')}")
print(f"  Compatibility Mode: {info.get('compatibility_mode', False)}")
print(f"  Migration Guide: {info.get('migration_guide', 'N/A')}")

# Example 3: Model Migration
print("\n" + "=" * 70)
print("Example 3: Model Migration")
print("=" * 70)

from ara.compat.migration import ModelMigrator
from pathlib import Path

print("\nModel Migration Tool:")
print("  Use ModelMigrator to migrate old models to new format")
print("  Example:")
print("    migrator = ModelMigrator(verbose=True)")
print("    summary = migrator.migrate_model_directory(")
print("        old_dir='models',")
print("        new_dir='.ara_cache/models',")
print("        backup=True")
print("    )")

# Example 4: Data Migration
print("\n" + "=" * 70)
print("Example 4: Data Migration")
print("=" * 70)

from ara.compat.migration import DataMigrator

print("\nData Migration Tool:")
print("  Use DataMigrator to migrate cache files")
print("  Example:")
print("    migrator = DataMigrator(verbose=True)")
print("    summary = migrator.migrate_cache_directory(")
print("        old_cache_dir='.ara_cache',")
print("        new_cache_dir='.ara_cache/market_data',")
print("        backup=True")
print("    )")

# Example 5: Deprecation Timeline
print("\n" + "=" * 70)
print("Example 5: Deprecation Timeline")
print("=" * 70)

from ara.compat.deprecation import DeprecationTimeline

print("\nDeprecation Schedule:")
print(f"  Current Version: {DeprecationTimeline.get_current_version()}")
print(f"  Removal Version: {DeprecationTimeline.get_removal_version()}")
print(f"  Removal Date: {DeprecationTimeline.get_removal_date().strftime('%B %Y')}")
print("\n  Timeline:")
print("    v4.0.0 (Jan 2024)  - Compatibility layer introduced")
print("    v4.5.0 (Jun 2024)  - Enhanced deprecation warnings")
print("    v5.0.0 (Jan 2025)  - Compatibility layer REMOVED")

# Example 6: Migration Guide
print("\n" + "=" * 70)
print("Example 6: Migration Guide")
print("=" * 70)

print("\nOld API (v3.x) - Deprecated:")
print("  from meridianalgo.core import predict_stock")
print("  result = predict_stock('AAPL', days=5)")

print("\nNew API (v4.0) - Recommended:")
print("  from ara.api.prediction_engine import PredictionEngine")
print("  import asyncio")
print("  ")
print("  async def main():")
print("      engine = PredictionEngine()")
print("      result = await engine.predict('AAPL', days=5)")
print("      return result")
print("  ")
print("  result = asyncio.run(main())")

print("\nFor detailed migration guide, see:")
print("  docs/MIGRATION_GUIDE.md")

# Example 7: Deprecation Warning Summary
print("\n" + "=" * 70)
print("Example 7: Deprecation Warning Summary")
print("=" * 70)

from ara.compat.deprecation import get_warning_manager

manager = get_warning_manager()
summary = manager.get_deprecation_summary()

print(f"\nDeprecation Warnings Issued: {summary['warnings_issued']}")
if summary['features']:
    print(f"Features Used:")
    for feature in summary['features']:
        print(f"  - {feature}")

print(f"\nMigration Guide Shown: {summary['migration_guide_shown']}")

# Final recommendations
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print("\n1. [OK] Use compatibility layer for quick fixes (temporary)")
print("2. [OK] Migrate models and data using migration tools")
print("3. [OK] Gradually refactor code to use new async API")
print("4. [OK] Complete migration before v5.0.0 (January 2025)")
print("5. [OK] Read migration guide: docs/MIGRATION_GUIDE.md")

print("\n" + "=" * 70)
print("Demo Complete")
print("=" * 70)
