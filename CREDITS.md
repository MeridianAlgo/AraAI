# Credits and Acknowledgments

This project builds upon the work of many talented individuals and organizations. We are grateful for their contributions to the open-source community.

---

## Core Development Team

### MeridianAlgo Team
**Primary Developers and Maintainers**

The ARA AI project is developed and maintained by the MeridianAlgo team, specializing in advanced algorithmic trading and machine learning solutions.

- **Project Lead**: MeridianAlgo Team
- **Core Development**: Python ML frameworks and trading algorithms
- **Repository**: https://github.com/MeridianAlgo
- **Contact**: support@meridianalgo.com

---

## Machine Learning Frameworks

### XGBoost
**Extreme Gradient Boosting Library**

XGBoost is used as our primary gradient boosting framework, achieving 99.7% accuracy in our ensemble.

- **Developers**: DMLC (Distributed Machine Learning Community)
- **Key Contributors**: Tianqi Chen, Carlos Guestrin
- **License**: Apache License 2.0
- **Repository**: https://github.com/dmlc/xgboost
- **Paper**: "XGBoost: A Scalable Tree Boosting System" (KDD 2016)

### LightGBM
**Light Gradient Boosting Machine**

LightGBM provides fast and efficient gradient boosting, contributing to our ensemble's performance.

- **Developer**: Microsoft Corporation
- **Key Contributors**: Guolin Ke, Qi Meng, Thomas Finley
- **License**: MIT License
- **Repository**: https://github.com/microsoft/LightGBM
- **Paper**: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (NIPS 2017)

### scikit-learn
**Machine Learning in Python**

scikit-learn provides the foundation for our Random Forest, Extra Trees, Gradient Boosting, Ridge, Elastic Net, and Lasso models.

- **Developers**: scikit-learn developers
- **Key Contributors**: David Cournapeau, Fabian Pedregosa, Gael Varoquaux, and many others
- **License**: BSD 3-Clause License
- **Repository**: https://github.com/scikit-learn/scikit-learn
- **Website**: https://scikit-learn.org

---

## Deep Learning and NLP

### PyTorch
**Deep Learning Framework**

PyTorch powers our neural network implementations and serves as the backend for Hugging Face transformers.

- **Developer**: Meta AI (Facebook AI Research)
- **Key Contributors**: Adam Paszke, Sam Gross, Soumith Chintala
- **License**: BSD-style License
- **Repository**: https://github.com/pytorch/pytorch
- **Website**: https://pytorch.org

### Hugging Face Transformers
**State-of-the-Art Natural Language Processing**

Hugging Face transformers provide our sentiment analysis and financial text analysis capabilities.

- **Developer**: Hugging Face Inc.
- **Key Contributors**: Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond
- **License**: Apache License 2.0
- **Repository**: https://github.com/huggingface/transformers
- **Website**: https://huggingface.co

#### Specific Models Used

**RoBERTa (Robustly Optimized BERT)**
- **Developers**: Facebook AI Research
- **Paper**: "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)
- **Model**: cardiffnlp/twitter-roberta-base-sentiment-latest
- **Use**: Market sentiment analysis

**FinBERT**
- **Developers**: ProsusAI
- **Paper**: "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"
- **Use**: Financial text analysis

---

## Data and APIs

### yfinance
**Yahoo Finance Market Data Downloader**

yfinance provides free access to historical and real-time market data.

- **Developer**: Ran Aroussi
- **License**: Apache License 2.0
- **Repository**: https://github.com/ranaroussi/yfinance
- **Note**: Unofficial Yahoo Finance API wrapper

### Yahoo Finance
**Market Data Provider**

Yahoo Finance provides the underlying market data accessed through yfinance.

- **Provider**: Yahoo Inc.
- **Website**: https://finance.yahoo.com
- **Note**: Data subject to Yahoo Finance Terms of Service

---

## Scientific Computing

### NumPy
**Numerical Computing Library**

NumPy provides the foundation for all numerical operations and array manipulations.

- **Developers**: NumPy Developers
- **Key Contributors**: Travis Oliphant, and many others
- **License**: BSD License
- **Repository**: https://github.com/numpy/numpy
- **Website**: https://numpy.org

### pandas
**Data Analysis and Manipulation**

pandas handles all data structures and time series operations.

- **Developer**: Wes McKinney and pandas development team
- **License**: BSD 3-Clause License
- **Repository**: https://github.com/pandas-dev/pandas
- **Website**: https://pandas.pydata.org

### SciPy
**Scientific Computing**

SciPy provides advanced mathematical functions and statistical operations.

- **Developers**: SciPy Developers
- **License**: BSD License
- **Repository**: https://github.com/scipy/scipy
- **Website**: https://scipy.org

---

## User Interface

### Rich
**Rich Text and Beautiful Formatting**

Rich provides our beautiful console output and formatting.

- **Developer**: Will McGugan
- **License**: MIT License
- **Repository**: https://github.com/Textualize/rich
- **Website**: https://rich.readthedocs.io

---

## Development Tools

### pytest
**Testing Framework**

pytest powers our comprehensive test suite.

- **Developers**: Holger Krekel and pytest-dev team
- **License**: MIT License
- **Repository**: https://github.com/pytest-dev/pytest
- **Website**: https://pytest.org

### Black
**Code Formatter**

Black ensures consistent code formatting across the project.

- **Developer**: Łukasz Langa and contributors
- **License**: MIT License
- **Repository**: https://github.com/psf/black
- **Website**: https://black.readthedocs.io

### Flake8
**Style Guide Enforcement**

Flake8 helps maintain code quality and PEP 8 compliance.

- **Developers**: Tarek Ziadé and contributors
- **License**: MIT License
- **Repository**: https://github.com/PyCQA/flake8
- **Website**: https://flake8.pycqa.org

---

## Infrastructure

### Docker
**Containerization Platform**

Docker enables consistent development and deployment environments.

- **Developer**: Docker, Inc.
- **License**: Apache License 2.0
- **Website**: https://www.docker.com

### GitHub Actions
**CI/CD Platform**

GitHub Actions powers our continuous integration and deployment pipeline.

- **Provider**: GitHub, Inc. (Microsoft)
- **Website**: https://github.com/features/actions

---

## Research and Academic Contributions

### Academic Papers

This project implements concepts from numerous academic papers:

1. **Gradient Boosting**
   - Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine"

2. **Random Forests**
   - Breiman, L. (2001). "Random Forests"

3. **BERT and Transformers**
   - Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
   - Vaswani, A., et al. (2017). "Attention Is All You Need"

4. **Financial Machine Learning**
   - López de Prado, M. (2018). "Advances in Financial Machine Learning"

---

## Community Contributors

We thank all contributors who have:
- Reported bugs and issues
- Suggested features and improvements
- Submitted pull requests
- Provided feedback and testing
- Contributed to documentation

### How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

---

## Special Thanks

### Open Source Community

We are grateful to the entire open-source community for creating and maintaining the tools and libraries that make this project possible.

### Financial Data Providers

Thanks to Yahoo Finance and other data providers for making financial data accessible for research and educational purposes.

### Academic Researchers

Thanks to researchers worldwide who publish their work openly, advancing the field of machine learning and financial analysis.

---

## Disclaimer

While we acknowledge and credit the creators of the tools and models we use, this does not imply endorsement of our project by these individuals or organizations.

All trademarks and registered trademarks are the property of their respective owners.

---

## License Compliance

This project complies with all licenses of the dependencies used:

- **MIT License**: Rich, pytest, Black, Flake8, LightGBM, and others
- **BSD License**: scikit-learn, NumPy, pandas, SciPy, PyTorch
- **Apache License 2.0**: XGBoost, Hugging Face Transformers, yfinance

For complete license information, see the [LICENSE](LICENSE) file and individual package licenses.

---

## Updates

This credits file is maintained and updated regularly. If you believe you should be credited or if there are any errors, please contact us or submit a pull request.

**Last Updated**: September 21, 2025  
**Version**: 2.2.0-Beta

---

**Thank you to everyone who has contributed to making this project possible!**
