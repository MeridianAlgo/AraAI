#!/usr/bin/env python3
"""
Install Ultimate ML Requirements including Hugging Face models
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    """Install all required packages for ultimate ML system"""
    print("🚀 Installing ULTIMATE ML Requirements")
    print("=" * 50)
    
    # Core ML packages (enhanced versions)
    core_packages = [
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0", 
        "lightgbm>=3.3.0",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "yfinance>=0.2.0",
        "rich>=13.0.0",
        "joblib>=1.2.0",
        "pytz>=2022.1"
    ]
    
    # Hugging Face and AI packages
    ai_packages = [
        "transformers>=4.21.0",
        "torch>=2.0.0",
        "tokenizers>=0.13.0",
        "datasets>=2.0.0",
        "accelerate>=0.20.0"
    ]
    
    # Financial data packages
    finance_packages = [
        "beautifulsoup4>=4.11.0",
        "requests>=2.28.0",
        "lxml>=4.9.0",
        "html5lib>=1.1"
    ]
    
    # Visualization and analysis
    viz_packages = [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "tqdm>=4.64.0"
    ]
    
    all_packages = core_packages + ai_packages + finance_packages + viz_packages
    
    success_count = 0
    total_packages = len(all_packages)
    
    # Install core packages
    print("\n📦 Installing Core ML Packages:")
    print("-" * 30)
    
    for package in core_packages:
        if install_package(package):
            success_count += 1
    
    # Install AI packages
    print("\n🤖 Installing Hugging Face AI Packages:")
    print("-" * 30)
    
    for package in ai_packages:
        if install_package(package):
            success_count += 1
    
    # Install finance packages
    print("\n💰 Installing Financial Data Packages:")
    print("-" * 30)
    
    for package in finance_packages:
        if install_package(package):
            success_count += 1
    
    # Install visualization packages
    print("\n📊 Installing Visualization Packages:")
    print("-" * 30)
    
    for package in viz_packages:
        if install_package(package):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Installation Summary:")
    print(f"✓ Successfully installed: {success_count}/{total_packages} packages")
    
    if success_count == total_packages:
        print("🎉 All packages installed successfully!")
        print("\n🚀 You can now run the ULTIMATE system:")
        print("  python ara_fast.py AAPL --verbose")
        print("  python ara_fast.py TSLA --retrain")
    else:
        failed_count = total_packages - success_count
        print(f"⚠️  {failed_count} packages failed to install")
        print("You may need to install them manually")
    
    # Test critical imports
    print("\n🧪 Testing critical imports...")
    test_ultimate_imports()

def test_ultimate_imports():
    """Test if ultimate packages can be imported"""
    test_packages = [
        ('sklearn', 'scikit-learn'),
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm'),
        ('transformers', 'transformers'),
        ('torch', 'torch'),
        ('yfinance', 'yfinance'),
        ('rich', 'rich'),
        ('joblib', 'joblib'),
        ('pytz', 'pytz')
    ]
    
    for import_name, package_name in test_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name} import successful")
        except ImportError as e:
            print(f"✗ {package_name} import failed: {e}")
    
    # Test Hugging Face models
    try:
        from transformers import pipeline
        print("✓ Hugging Face transformers ready")
        
        # Test if we can load a simple model
        try:
            sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            print("✓ Hugging Face models can be loaded")
        except Exception as e:
            print(f"⚠️ Hugging Face model loading may require internet: {e}")
            
    except ImportError:
        print("✗ Hugging Face transformers not available")

if __name__ == "__main__":
    main()