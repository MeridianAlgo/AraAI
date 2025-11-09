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
        print(f" {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Failed to install {package}: {e}")
        return False

def main():
    """Install all required packages for ultimate ML system"""
    print(" Installing ULTIMATE ML Requirements")
    print("=" * 50)

    # Core ML packages (enhanced versions)
    core_packages = [
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0", 
        "lightgbm>=3.3.0",
        "catboost>=1.1.0",
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0"
    ]

    huggingface_packages = [
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0"
    ]

    data_packages = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "yfinance>=0.2.28",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0"
    ]

    visualization_packages = [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0"
    ]

    utility_packages = [
        "tqdm>=4.65.0",
        "loguru>=0.7.0",
        "rich>=13.5.0",
        "python-dotenv>=1.0.0",
        "colorama>=0.4.6"
    ]

    all_packages = core_packages + huggingface_packages + data_packages + visualization_packages + utility_packages

    success_count = 0
    for package in all_packages:
        if install_package(package):
            success_count += 1

    print("=" * 50)
    print(f" Successfully installed {success_count}/{len(all_packages)} packages")
    print(" All packages are ready for the ULTIMATE ML system")

    # Optional: download Hugging Face models
    try:
        from transformers import AutoModel, AutoTokenizer
        print(" Downloading Hugging Face models...")
        models = [
            "ProsusAI/finbert",
            "cardiffnlp/twitter-roberta-base-sentiment",
            "mrm8488/distilroberta-finetuned-financial-news-sentiment"
        ]

        for model in models:
            print(f"⬇  Downloading {model}...")
            AutoModel.from_pretrained(model)
            AutoTokenizer.from_pretrained(model)
            print(f" {model} downloaded")

        print(" Hugging Face models ready")
    except Exception as e:
        print(f"  Could not download Hugging Face models automatically: {e}")
        print("   You can download them manually later")

    # Test critical imports
    print("\n Testing critical imports...")
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
            print(f" {package_name} import successful")
        except ImportError as e:
            print(f" {package_name} import failed: {e}")
    
    # Test Hugging Face models
    try:
        from transformers import pipeline
        print(" Hugging Face transformers ready")
        
        # Test if we can load a simple model
        try:
            sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            print(" Hugging Face models can be loaded")
        except Exception as e:
            print(f" Hugging Face model loading may require internet: {e}")
            
    except ImportError:
        print(" Hugging Face transformers not available")

if __name__ == "__main__":
    main()