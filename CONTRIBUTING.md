# Contributing to ARA AI

Thank you for your interest in contributing to ARA AI! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)
- [Documentation](#documentation)

## 🤝 Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of machine learning and stock analysis

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/YOUR_USERNAME/AraAI.git
cd AraAI
```

3. Add upstream remote:
```bash
git remote add upstream https://github.com/MeridianAlgo/AraAI.git
```

## 💻 Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
# Test imports
python -c "import meridianalgo; print(f'Version: {meridianalgo.__version__}')"

# Run tests
pytest tests/ -v
```

## 🔨 Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style
- Add comments for complex logic
- Update documentation as needed

### 3. Add Tests

- Add unit tests for new features
- Ensure existing tests still pass
- Aim for high code coverage

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Tests

```bash
pytest tests/test_ultimate_ml.py -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=meridianalgo --cov-report=html
```

### Test on Multiple Python Versions

```bash
# Using tox (if configured)
tox

# Or manually
python3.9 -m pytest tests/
python3.10 -m pytest tests/
python3.11 -m pytest tests/
```

## 📤 Submitting Changes

### 1. Commit Your Changes

```bash
git add .
git commit -m "feat: Add new feature description"
```

**Commit Message Format:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `style:` Code style changes
- `perf:` Performance improvements

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template:
- Description of changes
- Related issues
- Testing performed
- Screenshots (if applicable)

### 4. Code Review

- Address review comments
- Update your PR as needed
- Be responsive to feedback

## 📝 Style Guidelines

### Python Code Style

- Follow PEP 8
- Use Black for formatting:
```bash
black meridianalgo/ ara.py ara_fast.py
```

- Use isort for imports:
```bash
isort meridianalgo/ ara.py ara_fast.py
```

- Run Flake8:
```bash
flake8 meridianalgo/ --max-line-length=120
```

### Code Quality

- Write docstrings for all functions/classes
- Use type hints where appropriate
- Keep functions focused and small
- Avoid deep nesting
- Use meaningful variable names

### Example

```python
def calculate_prediction(
data: pd.DataFrame,
days: int = 5
) -> Dict[str, Any]:
"""
Calculate stock price predictions.

Args:
data: Historical stock data
days: Number of days to predict

Returns:
Dictionary containing predictions and metadata
"""
# Implementation
pass
```

## 📚 Documentation

### Update Documentation

- Update README.md for user-facing changes
- Update docstrings for code changes
- Add examples for new features
- Update CHANGELOG.md

### Documentation Files

- `README.md` - Main project documentation
- `docs/INSTALLATION.md` - Installation guide
- `docs/QUICK_START.md` - Quick start guide
- `docs/USER_MANUAL.md` - Detailed user manual
- `docs/TROUBLESHOOTING.md` - Common issues and solutions
- `docs/CI_CD_SETUP.md` - CI/CD documentation

## 🐛 Reporting Bugs

### Before Reporting

1. Check existing issues
2. Test with latest version
3. Verify it's reproducible

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Run command...
2. See error...

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python Version: [e.g., 3.11.5]
- ARA AI Version: [e.g., 2.2.0-Beta]

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other approaches you've thought about

**Additional Context**
Any other relevant information
```

## 🔍 Code Review Process

### What We Look For

- Code quality and readability
- Test coverage
- Documentation updates
- Performance impact
- Security considerations
- Backward compatibility

### Review Timeline

- Initial review: Within 3-5 days
- Follow-up reviews: Within 2-3 days
- Merge: After approval and CI/CD passes

## 🎯 Areas for Contribution

### High Priority

- Bug fixes
- Performance improvements
- Test coverage improvements
- Documentation enhancements

### Medium Priority

- New ML models
- Additional technical indicators
- UI/UX improvements
- Platform-specific optimizations

### Low Priority

- Code refactoring
- Style improvements
- Example additions

## 📞 Getting Help

### Resources

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/AraAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeridianAlgo/AraAI/discussions)

### Contact

- Open an issue for bugs or features
- Start a discussion for questions
- Email: support@meridianalgo.com

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to ARA AI!**

Your contributions help make stock prediction more accessible and accurate for everyone.


---

## Disclaimer

By contributing to this project, you agree that:

- Your contributions will be licensed under the MIT License
- You have the right to submit the contributions
- Your contributions are your own original work
- You understand this is open-source software provided AS IS
- You will not hold the project maintainers liable for any issues
- You have read and agree to follow the Code of Conduct

This project is for educational purposes. Contributors are not responsible for how users apply the software or any financial outcomes.

---

**Last Updated**: September 21, 2025
