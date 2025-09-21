#!/usr/bin/env python3
"""
MeridianAlgo - Advanced AI Stock Analysis Package
Enhanced with Ara AI's ensemble ML system, intelligent caching, and multi-GPU support
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Advanced AI Stock Analysis Package with ensemble ML models and intelligent prediction caching"

# Read requirements from requirements.txt
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "torch>=1.12.0",
            "scikit-learn>=1.1.0", 
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "yfinance>=0.1.87",
            "requests>=2.28.0",
            "rich>=12.0.0"
        ]

setup(
    name="meridianalgo",
    version="2.1.0",  # Major version bump for enhanced Ara AI integration
    author="MeridianAlgo Team",
    author_email="support@meridianalgo.com",
    description="Advanced AI Stock Analysis with Ensemble ML Models and Intelligent Caching",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/MeridianAlgo/Ara",
    project_urls={
        "Bug Tracker": "https://github.com/MeridianAlgo/Ara/issues",
        "Documentation": "https://github.com/MeridianAlgo/Ara/blob/main/README.md",
        "Source Code": "https://github.com/MeridianAlgo/Ara",
        "PyPI": "https://pypi.org/project/meridianalgo/"
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "gpu-nvidia": ["torch[cuda]"],
        "gpu-amd": ["torch-directml"],
        "gpu-intel": ["intel-extension-for-pytorch"],
        "performance": ["numba>=0.56.0", "joblib>=1.2.0"],
        "dev": ["pytest>=7.0.0", "flake8>=5.0.0", "black>=22.0.0"],
    },
    entry_points={
        "console_scripts": [
            "ara=meridianalgo.cli:main",
            "meridianalgo=meridianalgo.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "meridianalgo": ["*.md", "*.txt", "*.json"],
    },
    keywords=[
        "stock analysis", "machine learning", "AI", "financial prediction",
        "ensemble models", "LSTM", "random forest", "gradient boosting",
        "technical indicators", "market prediction", "algorithmic trading",
        "GPU acceleration", "intelligent caching", "prediction accuracy"
    ],
    zip_safe=False,
)