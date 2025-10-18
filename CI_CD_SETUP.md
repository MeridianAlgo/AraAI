# 🚀 CI/CD Pipeline Setup Guide

## Overview

This document describes the comprehensive CI/CD pipeline for ARA AI, ensuring quality, reliability, and cross-platform compatibility.

## 🎯 Pipeline Features

### ✅ **Cross-Platform Testing**
- **Linux** (Ubuntu Latest)
- **Windows** (Windows Latest)
- **macOS** (macOS Latest)

### ✅ **Multi-Version Python Support**
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12

### ✅ **Comprehensive Testing**
- Unit tests
- Integration tests
- Performance benchmarks
- Security scanning
- Code quality checks

### ✅ **Automated Releases**
- GitHub releases
- PyPI publishing
- Platform-specific binaries
- Documentation updates

## 📋 Prerequisites

### For Local Development
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 pylint

# Or use the dev extras
pip install -e ".[dev]"
```

### For Docker Testing
```bash
# Install Docker and Docker Compose
# https://docs.docker.com/get-docker/

# Build and test
docker-compose up ara-test
```

## 🔧 Setup Instructions

### 1. GitHub Repository Setup

1. **Enable GitHub Actions**
   - Go to repository Settings → Actions → General
   - Enable "Allow all actions and reusable workflows"

2. **Configure Branch Protection** (Optional but recommended)
   - Go to Settings → Branches
   - Add rule for `main` branch:
     - Require status checks to pass
     - Require branches to be up to date
     - Include administrators

3. **Add Secrets** (if needed)
   - Go to Settings → Secrets and variables → Actions
   - Add `PYPI_API_TOKEN` for PyPI publishing
   - Add `CODECOV_TOKEN` for coverage reporting

### 2. Local Testing Setup

```bash
# Clone the repository
git clone https://github.com/MeridianAlgo/AraAI.git
cd AraAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### 3. Docker Setup

```bash
# Build development image
docker-compose build ara-dev

# Run tests in Docker
docker-compose up ara-test

# Run production build
docker-compose up ara-prod
```

## 🧪 Running Tests

### Local Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=meridianalgo --cov-report=html

# Run specific test file
pytest tests/test_ultimate_ml.py -v

# Run tests in parallel
pytest tests/ -v -n auto

# Run only fast tests
pytest tests/ -v -m "not slow"
```

### Docker Testing

```bash
# Run full test suite
docker-compose up ara-test

# Run specific tests
docker-compose run ara-test pytest tests/test_ultimate_ml.py -v

# Interactive testing
docker-compose run ara-test bash
```

### Platform-Specific Testing

```bash
# Test on Linux
docker-compose run --rm ara-test

# Test on Windows (use Windows Docker)
docker-compose -f docker-compose.windows.yml up ara-test

# Test on macOS (native)
pytest tests/ -v
```

## 📊 Code Quality Checks

### Formatting
```bash
# Check formatting
black --check meridianalgo/ ara.py ara_fast.py

# Auto-format
black meridianalgo/ ara.py ara_fast.py
```

### Linting
```bash
# Flake8
flake8 meridianalgo/ --max-line-length=120

# Pylint
pylint meridianalgo/ --disable=C0111,C0103
```

### Import Sorting
```bash
# Check imports
isort --check-only meridianalgo/

# Fix imports
isort meridianalgo/
```

### Type Checking
```bash
# MyPy (if configured)
mypy meridianalgo/
```

## 🔒 Security Scanning

### Dependency Security
```bash
# Install safety
pip install safety

# Check dependencies
safety check

# Check with detailed output
safety check --json
```

### Code Security
```bash
# Install bandit
pip install bandit

# Scan code
bandit -r meridianalgo/

# Generate report
bandit -r meridianalgo/ -f json -o security-report.json
```

## 📦 Building Releases

### Manual Release Process

1. **Update Version**
   ```bash
   # Update version in meridianalgo/__init__.py
   __version__ = "2.2.0-Beta"
   ```

2. **Update Changelog**
   ```bash
   # Edit CHANGELOG.md with new features
   ```

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "Release v2.2.0-Beta"
   ```

4. **Create Tag**
   ```bash
   git tag -a v2.2.0-Beta -m "Release v2.2.0-Beta"
   ```

5. **Push to GitHub**
   ```bash
   git push origin main
   git push origin v2.2.0-Beta
   ```

6. **GitHub Actions will automatically:**
   - Run all tests
   - Build release assets
   - Create GitHub release
   - Publish to PyPI (if configured)

### Automated Release (via CI/CD)

Simply push a tag:
```bash
git tag -a v2.2.0-Beta -m "Release v2.2.0-Beta"
git push origin v2.2.0-Beta
```

## 🐛 Troubleshooting

### Tests Failing Locally

1. **Check Python version**
   ```bash
   python --version
   ```

2. **Update dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Clear cache**
   ```bash
   pytest --cache-clear
   rm -rf .pytest_cache __pycache__
   ```

### CI/CD Failures

1. **Check GitHub Actions logs**
   - Go to Actions tab in GitHub
   - Click on failed workflow
   - Review error messages

2. **Test locally with same Python version**
   ```bash
   pyenv install 3.11
   pyenv local 3.11
   pytest tests/ -v
   ```

3. **Test in Docker (matches CI environment)**
   ```bash
   docker-compose up ara-test
   ```

### Docker Issues

1. **Rebuild images**
   ```bash
   docker-compose build --no-cache
   ```

2. **Clean up**
   ```bash
   docker-compose down -v
   docker system prune -a
   ```

3. **Check logs**
   ```bash
   docker-compose logs ara-test
   ```

## 📈 Performance Optimization

### Speed Up CI/CD

1. **Use caching**
   - Dependencies are cached automatically
   - Models can be cached between runs

2. **Parallelize tests**
   ```yaml
   # In GitHub Actions
   strategy:
     matrix:
       python-version: [3.9, 3.11]  # Reduce versions
   ```

3. **Skip slow tests in PR**
   ```bash
   pytest tests/ -v -m "not slow"
   ```

### Reduce Test Time

1. **Use pytest-xdist**
   ```bash
   pytest tests/ -v -n auto
   ```

2. **Mark slow tests**
   ```python
   @pytest.mark.slow
   def test_long_running():
       pass
   ```

3. **Use fixtures efficiently**
   ```python
   @pytest.fixture(scope="session")
   def ml_system():
       return UltimateStockML()
   ```

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pytest Documentation](https://docs.pytest.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Python Packaging Guide](https://packaging.python.org/)

## 🤝 Contributing

When contributing to ARA AI:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests locally
5. Submit a pull request
6. CI/CD will automatically test your changes

## 📞 Support

For CI/CD issues:
- Open an issue on GitHub
- Check existing issues and discussions
- Review GitHub Actions logs

---

**Last Updated**: September 21, 2025  
**Version**: 2.2.0-Beta
