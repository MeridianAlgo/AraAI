#!/usr/bin/env python3
"""
Setup script for MeridianAlgo Smart Trader
Ultra-Accurate AI Stock Analysis with Universal GPU Support
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="meridianalgo-smarttrader",
    version="1.0.1",
    author="MeridianAlgo",
    author_email="contact@meridianalgo.com",
    description="Ultra-Accurate AI Stock Analysis with Universal GPU Support (AMD • Intel • NVIDIA • Apple)",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/MeridianAlgo/In-Python",
    project_urls={
        "Bug Tracker": "https://github.com/MeridianAlgo/In-Python/issues",
        "Documentation": "https://github.com/MeridianAlgo/In-Python/blob/main/README.md",
        "Source Code": "https://github.com/MeridianAlgo/In-Python",
        "PyPI": "https://pypi.org/project/meridianalgo/",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
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
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: GPU",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "gpu-nvidia": ["torch[cuda]", "torchvision", "torchaudio"],
        "gpu-amd": ["torch-directml"],
        "gpu-intel": ["intel-extension-for-pytorch"],
        "dev": ["pytest", "black", "flake8", "mypy"],
        "all": ["torch[cuda]", "torchvision", "torchaudio", "torch-directml", "intel-extension-for-pytorch"],
    },
    entry_points={
        "console_scripts": [
            "smart-trader=meridianalgo.smarttrader.cli:main",
            "smarttrader=meridianalgo.smarttrader.cli:main",
        ],
    },
    keywords=[
        "stock-analysis", "ai", "machine-learning", "trading", "finance", 
        "gpu", "amd", "nvidia", "intel", "apple-silicon", "pytorch",
        "ensemble-learning", "volatility", "technical-analysis", "predictions"
    ],
    include_package_data=True,
    zip_safe=False,
)