#!/usr/bin/env python3
"""
Enhanced Ara AI Stock Analysis Platform
Improved error handling, directory management, and security
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
import configparser

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class AraDirectoryManager:
    """Manages Ara AI directory structure and configuration"""
    
    def __init__(self):
        self.setup_directories()
        self.setup_logging()
        self.load_config()
    
    def setup_directories(self):
        """Create secure directory structure"""
        try:
            # Use user's home directory for cross-platform compatibility
            if sys.platform == "win32":
                self.base_dir = Path.home() / "Documents" / "AraAI"
            else:
                self.base_dir = Path.home() / "AraAI"
            
            # Create directory structure
            self.data_dir = self.base_dir / "data"
            self.cache_dir = self.base_dir / "cache"
            self.logs_dir = self.base_dir / "logs"
            self.config_dir = self.base_dir / "config"
            self.models_dir = self.base_dir / "models"
            
            # Create all directories with proper permissions
            for directory in [self.base_dir, self.data_dir, self.cache_dir, 
                            self.logs_dir, self.config_dir, self.models_dir]:
                directory.mkdir(parents=True, exist_ok=True)
                
                # Set secure permissions (owner only)
                if sys.platform != "win32":
                    os.chmod(directory, 0o700)
            
            print(f"✅ Directory structure created at: {self.base_dir}")
            
        except Exception as e:
            print(f"❌ Error creating directories: {e}")
            # Fallback to current directory
            self.base_dir = Path.cwd() / "ara_data"
            self.base_dir.mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        try:
            log_file = self.logs_dir / f"ara_{datetime.now().strftime('%Y%m%d')}.log"
            
            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            
            self.logger = logging.getLogger('AraAI')
            self.logger.info("Logging system initialized")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not setup logging: {e}")
            self.logger = logging.getLogger('AraAI')
    
    def load_config(self):
        """Load configuration with secure defaults"""
        try:
            self.config_file = self.config_dir / "ara_config.ini"
            self.config = configparser.ConfigParser()
            
            # Default configuration
            default_config = {
                'DEFAULT': {
                    'data_dir': str(self.data_dir),
                    'cache_dir': str(self.cache_dir),
                    'logs_dir': str(self.logs_dir),
                    'verbose': 'false',
                    'cache_enabled': 'true',
                    'max_cache_age_hours': '24',
                    'max_predictions_per_symbol': '100',
                    'enable_gpu': 'true',
                    'security_mode': 'strict'
                }
            }
            
            # Create config file if it doesn't exist
            if not self.config_file.exists():
                self.config.read_dict(default_config)
                with open(self.config_file, 'w') as f:
                    self.config.write(f)
                
                # Set secure permissions
                if sys.platform != "win32":
                    os.chmod(self.config_file, 0o600)
                
                self.logger.info(f"Created configuration file: {self.config_file}")
            else:
                self.config.read(self.config_file)
                self.logger.info(f"Loaded configuration from: {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            # Use defaults
            self.config = configparser.ConfigParser()
            self.config.read_dict({'DEFAULT': {}})

class EnhancedAraAI:
    """Enhanced Ara AI with improved error handling and security"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.dir_manager = AraDirectoryManager()
        self.logger = self.dir_manager.logger
        
        try:
            # Import MeridianAlgo components with error handling
            self._import_components()
            self._initialize_system()
            
        except ImportError as e:
            self.logger.error(f"Failed to import MeridianAlgo: {e}")
            self._handle_import_error()
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise
    
    def _import_components(self):
        """Import MeridianAlgo components with error handling"""
        try:
            from meridianalgo import AraAI, quick_predict, analyze_accuracy
            from meridianalgo.utils import GPUManager, CacheManager, AccuracyTracker
            from meridianalgo.console import ConsoleManager
            
            self.AraAI = AraAI
            self.quick_predict = quick_predict
            self.analyze_accuracy = analyze_accuracy
            self.GPUManager = GPUManager
            self.CacheManager = CacheManager
            self.AccuracyTracker = AccuracyTracker
            self.ConsoleManager = ConsoleManager
            
            self.logger.info("Successfully imported MeridianAlgo components")
            
        except ImportError as e:
            self.logger.error(f"MeridianAlgo import failed: {e}")
            raise ImportError(
                "MeridianAlgo package not found. Please install it with:\n"
                "pip install meridianalgo"
            )
    
    def _initialize_system(self):
        """Initialize the enhanced Ara AI system"""
        try:
            # Initialize console manager
            self.console = self.ConsoleManager(verbose=self.verbose)
            
            # Initialize core Ara AI system
            self.ara = self.AraAI(verbose=self.verbose)
            
            # Initialize utility managers
            self.gpu_manager = self.GPUManager()
            self.cache_manager = self.CacheManager()
            self.accuracy_tracker = self.AccuracyTracker()
            
            # Override cache directory
            self.cache_manager.cache_file = str(self.dir_manager.cache_dir / "predictions.csv")
            self.accuracy_tracker.accuracy_file = str(self.dir_manager.cache_dir / "accuracy.csv")
            
            self.logger.info("Enhanced Ara AI system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise
    
    def _handle_import_error(self):
        """Handle MeridianAlgo import errors gracefully"""
        print("\n❌ MeridianAlgo package not found!")
        print("\n📦 To install MeridianAlgo:")
        print("   pip install meridianalgo")
        print("\n🔗 Or visit: https://pypi.org/project/meridianalgo/")
        print("\n💡 Alternative: Use the standalone ara.py file")
        sys.exit(1)
    
    def predict_with_error_handling(self, symbol, days=5, use_cache=True):
        """Enhanced prediction with comprehensive error handling"""
        try:
            self.logger.info(f"Starting prediction for {symbol} ({days} days)")
            
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                raise ValueError("Symbol must be a non-empty string")
            
            if not isinstance(days, int) or days < 1 or days > 30:
                raise ValueError("Days must be an integer between 1 and 30")
            
            symbol = symbol.upper().strip()
            
            # Check cache first if enabled
            if use_cache and self.dir_manager.config.getboolean('DEFAULT', 'cache_enabled', fallback=True):
                cached_result = self._check_cache_safely(symbol, days)
                if cached_result:
                    self.logger.info(f"Using cached prediction for {symbol}")
                    return cached_result
            
            # Generate new prediction
            result = self.ara.predict(symbol, days=days, use_cache=use_cache)
            
            if result:
                self.logger.info(f"Successfully generated prediction for {symbol}")
                self._save_prediction_safely(symbol, result)
                return result
            else:
                self.logger.warning(f"No prediction result for {symbol}")
                return None
                
        except ValueError as e:
            self.logger.error(f"Input validation error: {e}")
            self.console.print_error(f"Invalid input: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Prediction error for {symbol}: {e}")
            self.console.print_error(f"Prediction failed: {e}")
            return None
    
    def _check_cache_safely(self, symbol, days):
        """Safely check cache with error handling"""
        try:
            return self.cache_manager.check_cached_predictions(symbol, days)
        except Exception as e:
            self.logger.warning(f"Cache check failed: {e}")
            return None
    
    def _save_prediction_safely(self, symbol, result):
        """Safely save prediction with error handling"""
        try:
            self.cache_manager.save_predictions(symbol, result)
        except Exception as e:
            self.logger.warning(f"Failed to save prediction to cache: {e}")
    
    def get_system_status(self):
        """Get comprehensive system status"""
        try:
            status = {
                'directories': {
                    'base': str(self.dir_manager.base_dir),
                    'data': str(self.dir_manager.data_dir),
                    'cache': str(self.dir_manager.cache_dir),
                    'logs': str(self.dir_manager.logs_dir),
                    'config': str(self.dir_manager.config_dir)
                },
                'system': self.ara.get_system_info() if hasattr(self, 'ara') else {},
                'config': dict(self.dir_manager.config['DEFAULT']),
                'security': {
                    'secure_directories': sys.platform != "win32",
                    'config_protected': True,
                    'logging_enabled': True
                }
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def validate_system(self):
        """Validate system integrity"""
        try:
            issues = []
            
            # Check directories
            for name, path in [
                ('base', self.dir_manager.base_dir),
                ('data', self.dir_manager.data_dir),
                ('cache', self.dir_manager.cache_dir),
                ('logs', self.dir_manager.logs_dir),
                ('config', self.dir_manager.config_dir)
            ]:
                if not path.exists():
                    issues.append(f"Missing directory: {name} ({path})")
                elif not os.access(path, os.R_OK | os.W_OK):
                    issues.append(f"No read/write access: {name} ({path})")
            
            # Check MeridianAlgo
            try:
                import meridianalgo
                version = getattr(meridianalgo, '__version__', 'unknown')
                self.logger.info(f"MeridianAlgo version: {version}")
            except ImportError:
                issues.append("MeridianAlgo package not installed")
            
            # Check GPU availability
            try:
                gpu_info = self.gpu_manager.detect_gpu_vendor()
                if not any(gpu_info.values()):
                    self.logger.info("No GPU acceleration available (using CPU)")
            except Exception as e:
                issues.append(f"GPU detection failed: {e}")
            
            if issues:
                self.logger.warning(f"System validation found {len(issues)} issues")
                for issue in issues:
                    self.logger.warning(f"  - {issue}")
                return False, issues
            else:
                self.logger.info("System validation passed")
                return True, []
                
        except Exception as e:
            self.logger.error(f"System validation failed: {e}")
            return False, [f"Validation error: {e}"]

def main():
    """Enhanced main function with comprehensive error handling"""
    try:
        print("🚀 Enhanced Ara AI Stock Analysis Platform")
        print("=" * 50)
        
        # Initialize enhanced system
        ara = EnhancedAraAI(verbose=True)
        
        # Validate system
        is_valid, issues = ara.validate_system()
        if not is_valid:
            print("\n⚠️  System validation issues found:")
            for issue in issues:
                print(f"   - {issue}")
            print("\nContinuing with available functionality...")
        
        # Show system status
        status = ara.get_system_status()
        print(f"\n📁 Base Directory: {status['directories']['base']}")
        
        # Interactive mode
        print("\n💡 Usage Examples:")
        print("   ara.predict_with_error_handling('AAPL', days=5)")
        print("   ara.get_system_status()")
        print("   ara.validate_system()")
        
        return ara
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    enhanced_ara = main()