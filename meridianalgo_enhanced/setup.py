from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="meridianalgo",
    version="0.2.1",
    author="MeridianAlgo",
    author_email="meridianalgo@gmail.com",
    description="A Python library for algorithmic trading and financial analysis with AI/ML capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MeridianAlgo/Packages",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "requests>=2.25.0",
        "python-dateutil>=2.8.0",
        "yfinance>=0.2.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "ml": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "torchaudio>=0.9.0",
        ],
        "visualization": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
        ],
    },
    keywords="trading, finance, machine learning, AI, stock analysis, algorithmic trading, backtesting",
    project_urls={
        "Bug Reports": "https://github.com/MeridianAlgo/Packages/issues",
        "Source": "https://github.com/MeridianAlgo/Packages",
        "Documentation": "https://github.com/MeridianAlgo/Packages/wiki",
    },
)