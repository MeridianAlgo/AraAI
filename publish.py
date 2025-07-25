#!/usr/bin/env python3
"""
Publishing script for MeridianAlgo Smart Trader
Handles GitHub and PyPI publishing
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"🔄 {description}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def clean_build():
    """Clean previous build artifacts"""
    print("🧹 Cleaning build artifacts...")
    
    dirs_to_clean = ['build', 'dist', '*.egg-info']
    for dir_pattern in dirs_to_clean:
        for path in Path('.').glob(dir_pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed: {path}")
            elif path.is_file():
                path.unlink()
                print(f"Removed: {path}")

def check_requirements():
    """Check if required tools are installed"""
    print("🔍 Checking requirements...")
    
    # Check git
    if not shutil.which('git'):
        print("❌ Git not found. Please install Git.")
        return False
    
    # Check Python modules
    try:
        import build
        import twine
        print("✅ All required tools are available")
        return True
    except ImportError as e:
        print(f"❌ Missing Python module: {e}")
        print("Install with: pip install build twine")
        return False

def build_package():
    """Build the Python package"""
    print("📦 Building package...")
    
    # Clean first
    clean_build()
    
    # Build package
    if not run_command("python -m build", "Building package"):
        return False
    
    # Check if files were created
    dist_files = list(Path('dist').glob('*'))
    if not dist_files:
        print("❌ No distribution files created")
        return False
    
    print(f"✅ Created {len(dist_files)} distribution files:")
    for file in dist_files:
        print(f"  - {file}")
    
    return True

def test_package():
    """Test the built package"""
    print("🧪 Testing package...")
    
    # Check package with twine
    if not run_command("twine check dist/*", "Checking package with twine"):
        return False
    
    return True

def publish_to_pypi(test=True):
    """Publish to PyPI (test or production)"""
    if test:
        print("🚀 Publishing to Test PyPI...")
        cmd = "twine upload --repository testpypi dist/*"
        description = "Publishing to Test PyPI"
    else:
        print("🚀 Publishing to PyPI...")
        cmd = "twine upload dist/*"
        description = "Publishing to PyPI"
    
    print("Note: You'll need to enter your PyPI credentials")
    
    if not run_command(cmd, description):
        return False
    
    if test:
        print("✅ Package published to Test PyPI!")
        print("Test installation with:")
        print("pip install --index-url https://test.pypi.org/simple/ meridianalgo-smarttrader")
    else:
        print("✅ Package published to PyPI!")
        print("Install with:")
        print("pip install meridianalgo-smarttrader")
    
    return True

def setup_git():
    """Setup git repository for GitHub"""
    print("📝 Setting up Git repository...")
    
    # Check if git repo exists
    if not Path('.git').exists():
        if not run_command("git init", "Initializing git repository"):
            return False
    
    # Add all files
    if not run_command("git add .", "Adding files to git"):
        return False
    
    # Check if there are changes to commit
    result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    if not result.stdout.strip():
        print("✅ No changes to commit")
        return True
    
    # Commit changes
    commit_message = "feat: Ultra-Accurate AI Stock Analysis with Universal GPU Support"
    if not run_command(f'git commit -m "{commit_message}"', "Committing changes"):
        return False
    
    return True

def publish_to_github():
    """Publish to GitHub"""
    print("🐙 Publishing to GitHub...")
    
    # Check if remote exists
    result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True)
    
    if "origin" not in result.stdout:
        print("❌ No GitHub remote configured")
        print("Please add your GitHub repository:")
        print("git remote add origin https://github.com/YourUsername/smart-trader.git")
        return False
    
    # Push to GitHub
    if not run_command("git push -u origin main", "Pushing to GitHub"):
        # Try master branch if main fails
        if not run_command("git push -u origin master", "Pushing to GitHub (master)"):
            return False
    
    print("✅ Code pushed to GitHub!")
    return True

def create_release_notes():
    """Create release notes"""
    version = "1.0.0"
    
    release_notes = f"""# Smart Trader v{version}

## 🚀 Features

### Ultra-Accurate AI Stock Analysis
- **Ensemble ML Models**: LSTM + Transformer + XGBoost
- **Advanced Technical Analysis**: 17+ indicators
- **Real-time Market Data**: Live price feeds
- **Confidence Scoring**: 70-88% realistic confidence ranges

### Universal GPU Support
- **🔴 AMD GPUs**: ROCm + DirectML support
- **🔵 Intel Arc GPUs**: XPU acceleration  
- **🟢 NVIDIA GPUs**: CUDA acceleration
- **🍎 Apple Silicon**: MPS optimization
- **💻 CPU Fallback**: Multi-threaded optimization

### Professional Features
- **Clean Interface**: Minimalistic, essential information
- **CSV Export**: Detailed prediction data
- **Performance Metrics**: Technical scores and accuracy tracking
- **Market Regime Detection**: Bullish/Bearish/Sideways identification

## 📦 Installation

```bash
pip install meridianalgo-smarttrader
```

## 🎯 Quick Start

```bash
# Analyze any stock
smart-trader AAPL --epochs 10

# Check GPU support
smart-trader --gpu-info
```

## 🔧 GPU Setup

### AMD GPUs
```bash
pip install torch-directml  # Windows
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6  # Linux
```

### NVIDIA GPUs
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Intel Arc GPUs
```bash
pip install intel-extension-for-pytorch
```

## 📈 Performance

- **Training Time**: 1-3 seconds (CPU), 0.3-1 seconds (GPU)
- **Accuracy**: Professional-grade predictions with confidence scoring
- **Universal**: Works on any hardware (CPU/GPU)

## 🎯 What's New in v{version}

- ✅ Fixed model confidence (no more 0.0%)
- ✅ Minimalistic colors (only for important info)
- ✅ Universal GPU support (AMD, Intel, NVIDIA, Apple)
- ✅ Enhanced prediction accuracy
- ✅ Professional-grade output
- ✅ Comprehensive documentation

---

**Made with ❤️ by MeridianAlgo**
"""
    
    with open('RELEASE_NOTES.md', 'w', encoding='utf-8') as f:
        f.write(release_notes)
    
    print("✅ Release notes created: RELEASE_NOTES.md")

def main():
    """Main publishing workflow"""
    print("🚀 MeridianAlgo Smart Trader Publishing Script")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create release notes
    create_release_notes()
    
    # Build package
    if not build_package():
        print("❌ Package build failed")
        sys.exit(1)
    
    # Test package
    if not test_package():
        print("❌ Package test failed")
        sys.exit(1)
    
    # Setup git
    if not setup_git():
        print("❌ Git setup failed")
        sys.exit(1)
    
    # Ask user what to publish
    print("\n📋 Publishing Options:")
    print("1. Test PyPI only")
    print("2. Production PyPI only")
    print("3. GitHub only")
    print("4. Both GitHub and Test PyPI")
    print("5. Both GitHub and Production PyPI")
    print("6. All (GitHub + Test PyPI + Production PyPI)")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    success = True
    
    if choice in ['1', '4', '6']:
        success &= publish_to_pypi(test=True)
    
    if choice in ['2', '5', '6']:
        success &= publish_to_pypi(test=False)
    
    if choice in ['3', '4', '5', '6']:
        success &= publish_to_github()
    
    if success:
        print("\n🎉 Publishing completed successfully!")
        print("\n📦 Package Information:")
        print("- Name: meridianalgo-smarttrader")
        print("- Version: 1.0.0")
        print("- PyPI: https://pypi.org/project/meridianalgo-smarttrader/")
        print("- GitHub: https://github.com/MeridianAlgo/smart-trader")
        print("\n🚀 Installation:")
        print("pip install meridianalgo-smarttrader")
    else:
        print("\n❌ Publishing failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()