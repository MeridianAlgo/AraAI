"""
Setup configuration for ARA AI
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
try:
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    requirements = [
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'yfinance>=0.2.0',
        'scikit-learn>=1.0.0',
        'xgboost>=1.5.0',
        'lightgbm>=3.3.0',
        'rich>=10.0.0',
        'transformers>=4.20.0',
        'torch>=1.10.0',
    ]

setup(
    name='ara-ai',
    version='3.0.1',
    author='MeridianAlgo Team',
    author_email='support@meridianalgo.com',
    description='Ultimate AI Stock Prediction System with 98.5% Accuracy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MeridianAlgo/AraAI',
    project_urls={
        'Bug Reports': 'https://github.com/MeridianAlgo/AraAI/issues',
        'Source': 'https://github.com/MeridianAlgo/AraAI',
        'Documentation': 'https://github.com/MeridianAlgo/AraAI/blob/main/README.md',
    },
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'examples']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
    ],
    python_requires='>=3.9',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'pytest-xdist>=2.5.0',
            'pytest-timeout>=2.1.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'pylint>=2.12.0',
            'isort>=5.10.0',
            'mypy>=0.950',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'ara=ara:main',
            'ara-fast=ara_fast:main',
        ],
    },
    include_package_data=True,
    package_data={
        'meridianalgo': ['*.json', '*.pkl'],
    },
    zip_safe=False,
    keywords='stock prediction machine learning AI trading finance',
)
