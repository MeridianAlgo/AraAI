#!/usr/bin/env python3
"""
Project Organization Script
Organizes Ara AI project into proper directory structure
"""

import os
import shutil
from pathlib import Path

def organize_project():
    """Organize project into proper directory structure"""
    
    print("🗂️  Organizing Ara AI Project Structure")
    print("=" * 50)
    
    # Define directory structure
    directories = {
        'src/': ['ara_enhanced.py'],
        'scripts/': ['install.bat', 'install.sh', 'install.py', 'install.ps1'],
        'docs/': ['INSTALLATION.md', 'SECURITY.md'],
        'tools/': ['build_package.py', 'deploy_package.py', 'update_package.py'],
        'examples/': [],
        'tests/': [],
        'config/': ['.env.example'],
        'legacy/': ['ara.py', 'run_ara.py', 'ara_launcher.py', 'Ara_AI_Launcher.bat', 'start_ara.bat'],
        'archive/': ['DEPLOYMENT_INSTRUCTIONS.md', 'FINAL_DEPLOYMENT_COMMANDS.md', 'PROJECT_STATUS.md']
    }
    
    # Create directories
    for directory in directories.keys():
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Move files to appropriate directories
    for directory, files in directories.items():
        for file in files:
            if os.path.exists(file):
                try:
                    destination = Path(directory) / file
                    if not destination.exists():
                        shutil.move(file, destination)
                        print(f"📁 Moved {file} → {directory}")
                except Exception as e:
                    print(f"⚠️  Could not move {file}: {e}")
    
    # Move old install files to scripts
    old_install_files = [
        'install.bat', 'install.sh', 'install.py', 'install.ps1', 'install_mac.command'
    ]
    
    for file in old_install_files:
        if os.path.exists(file) and not os.path.exists(f'scripts/{file}'):
            try:
                shutil.move(file, f'scripts/{file}')
                print(f"📁 Moved {file} → scripts/")
            except Exception as e:
                print(f"⚠️  Could not move {file}: {e}")
    
    # Create example files
    create_examples()
    
    # Create test files
    create_tests()
    
    # Update gitignore
    update_gitignore()
    
    print("\n✅ Project organization complete!")
    print("\n📁 New project structure:")
    print_directory_tree()

def create_examples():
    """Create example files"""
    
    # Basic usage example
    basic_example = '''#!/usr/bin/env python3
"""
Basic Ara AI Usage Example
"""

from meridianalgo import quick_predict, analyze_accuracy

def main():
    """Basic usage examples"""
    
    print("🚀 Ara AI Basic Usage Examples")
    print("=" * 40)
    
    # Quick prediction
    print("\\n1. Quick Prediction:")
    result = quick_predict('AAPL', days=5)
    if result:
        print(f"AAPL predictions: {len(result['predictions'])} days")
        for pred in result['predictions']:
            print(f"  Day {pred['day']}: ${pred['predicted_price']:.2f}")
    
    # Accuracy analysis
    print("\\n2. Accuracy Analysis:")
    accuracy = analyze_accuracy('AAPL')
    if accuracy:
        print(f"AAPL accuracy: {accuracy['accuracy_rate']:.1f}%")

if __name__ == "__main__":
    main()
'''
    
    with open('examples/basic_usage.py', 'w', encoding='utf-8') as f:
        f.write(basic_example)
    
    # Advanced example
    advanced_example = '''#!/usr/bin/env python3
"""
Advanced Ara AI Usage Example
"""

from meridianalgo import AraAI
from meridianalgo.utils import GPUManager, CacheManager, AccuracyTracker

def main():
    """Advanced usage examples"""
    
    print("🚀 Ara AI Advanced Usage Examples")
    print("=" * 40)
    
    # Initialize full system
    ara = AraAI(verbose=True)
    
    # Check system capabilities
    print("\\n1. System Information:")
    system_info = ara.get_system_info()
    print(f"Device: {system_info['device']}")
    print(f"GPU Info: {system_info.get('gpu_info', {})}")
    
    # Advanced prediction
    print("\\n2. Advanced Prediction:")
    result = ara.predict('TSLA', days=7, use_cache=True)
    if result:
        print(f"TSLA prediction generated with {len(result['predictions'])} days")
    
    # Component usage
    print("\\n3. Individual Components:")
    
    # GPU Manager
    gpu_manager = GPUManager()
    gpu_info = gpu_manager.detect_gpu_vendor()
    print(f"GPU Support: {gpu_info}")
    
    # Cache Manager
    cache_manager = CacheManager()
    cache_stats = cache_manager.get_cache_stats()
    print(f"Cache Stats: {cache_stats}")
    
    # Accuracy Tracker
    tracker = AccuracyTracker()
    accuracy_stats = tracker.get_accuracy_stats()
    print(f"Accuracy Stats: {accuracy_stats}")

if __name__ == "__main__":
    main()
'''
    
    with open('examples/advanced_usage.py', 'w', encoding='utf-8') as f:
        f.write(advanced_example)
    
    print("✅ Created example files")

def create_tests():
    """Create test files"""
    
    # Basic test
    basic_test = '''#!/usr/bin/env python3
"""
Basic tests for Ara AI
"""

import unittest
from meridianalgo import quick_predict, analyze_accuracy, get_version_info

class TestBasicFunctionality(unittest.TestCase):
    """Test basic Ara AI functionality"""
    
    def test_version_info(self):
        """Test version information"""
        version_info = get_version_info()
        self.assertIsInstance(version_info, dict)
        self.assertIn('version', version_info)
        self.assertIn('features', version_info)
    
    def test_quick_predict_input_validation(self):
        """Test input validation"""
        # Test invalid symbol
        result = quick_predict('', days=5)
        self.assertIsNone(result)
        
        # Test invalid days
        result = quick_predict('AAPL', days=0)
        self.assertIsNone(result)
        
        result = quick_predict('AAPL', days=50)
        self.assertIsNone(result)
    
    def test_quick_predict_valid_input(self):
        """Test valid prediction"""
        result = quick_predict('AAPL', days=3)
        if result:  # May fail due to network issues
            self.assertIsInstance(result, dict)
            self.assertIn('predictions', result)
            self.assertEqual(len(result['predictions']), 3)

if __name__ == '__main__':
    unittest.main()
'''
    
    with open('tests/test_basic.py', 'w', encoding='utf-8') as f:
        f.write(basic_test)
    
    # Test runner
    test_runner = '''#!/usr/bin/env python3
"""
Test runner for Ara AI
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_tests():
    """Run all tests"""
    
    print("🧪 Running Ara AI Tests")
    print("=" * 30)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\\n📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
'''
    
    with open('tests/run_tests.py', 'w', encoding='utf-8') as f:
        f.write(test_runner)
    
    print("✅ Created test files")

def update_gitignore():
    """Update .gitignore with proper exclusions"""
    
    gitignore_content = '''# Ara AI - Enhanced .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Ara AI specific
predictions.csv
prediction_accuracy.csv
online_learning_data.csv
*.log
cache/
data/
models/
temp/

# Build artifacts
*.tar.gz
*.whl
*.zip

# Security
*.key
*.pem
*.crt
config/*.ini
.env.*

# Temporary files
*.tmp
*.temp
*.bak
*.backup
'''
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print("✅ Updated .gitignore")

def print_directory_tree():
    """Print the new directory structure"""
    
    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        path = Path(directory)
        if not path.exists():
            return
        
        items = sorted(path.iterdir())
        for i, item in enumerate(items):
            if item.name.startswith('.'):
                continue
            
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and current_depth < max_depth - 1:
                extension = "    " if is_last else "│   "
                print_tree(item, prefix + extension, max_depth, current_depth + 1)
    
    print_tree(".")

if __name__ == "__main__":
    organize_project()