# CI/CD Pipeline Documentation

## Overview

ARA AI uses GitHub Actions for comprehensive CI/CD automation across all platforms (Linux, Windows, macOS) and Python versions (3.8-3.12).

## Workflows

### 1. CI/CD Pipeline (`ci-tests.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Daily scheduled runs at 2 AM UTC

**Jobs:**

#### Lint (Code Quality)
- **Platform:** Ubuntu Latest
- **Python:** 3.11
- **Tools:** Black, isort, Flake8, Pylint
- **Purpose:** Ensure code quality and style consistency

#### Test (Cross-Platform)
- **Platforms:** Ubuntu, Windows, macOS
- **Python Versions:** 3.8, 3.9, 3.10, 3.11, 3.12
- **Matrix Strategy:** 13 combinations (some excluded for efficiency)
- **Tests:**
  - Import tests
  - Unit tests with coverage
  - Basic prediction tests
- **Coverage:** Uploaded to Codecov for Ubuntu 3.11

#### Integration Tests
- **Platforms:** Ubuntu, Windows, macOS
- **Python:** 3.11
- **Tests:**
  - Ultimate ML system tests
  - Multi-stock predictions (AAPL, MSFT, GOOGL)
  - End-to-end workflows

#### Performance Benchmarks
- **Platform:** Ubuntu Latest
- **Python:** 3.11
- **Metrics:**
  - Initialization time
  - Training time (10 stocks)
  - Prediction time
  - Total execution time

#### Security Scanning
- **Platform:** Ubuntu Latest
- **Tools:**
  - Safety (dependency security)
  - Bandit (code security)
- **Output:** JSON reports uploaded as artifacts

#### Documentation Build
- **Platform:** Ubuntu Latest
- **Checks:**
  - Documentation file validation
  - README validation
  - Sphinx build (if configured)

#### Build and Package
- **Platform:** Ubuntu Latest
- **Output:**
  - Source distribution (`.tar.gz`)
  - Wheel distribution (`.whl`)
  - Validated with `twine check`
  - Uploaded as artifacts

### 2. Release Pipeline (`release.yml`)

**Triggers:**
- Tags matching `v*.*.*`, `v*.*.*-Beta`, `v*.*.*-Alpha`

**Jobs:**

#### Create Release
- Extracts version from tag
- Determines if pre-release (Beta/Alpha)
- Generates changelog from CHANGELOG.md
- Creates GitHub release with full description

#### Build Assets
- **Platforms:** Ubuntu, Windows, macOS
- **Output:**
  - Standalone executables (PyInstaller)
  - Platform-specific archives
  - Uploaded to GitHub release

#### Test Release
- **Platforms:** Ubuntu, Windows, macOS
- **Python Versions:** 3.9, 3.11
- **Tests:**
  - Installation verification
  - Import tests
  - Quick functionality tests

#### Publish to PyPI (Optional)
- **Condition:** Only for stable releases (no Beta/Alpha)
- **Requirements:** `PYPI_API_TOKEN` secret
- **Output:** Package published to PyPI

#### Update Documentation
- Updates version in documentation
- Commits and pushes changes

## Running Tests Locally

### Using pytest
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=meridianalgo --cov-report=html

# Run specific test file
pytest tests/test_ultimate_ml.py -v

# Run specific test
pytest tests/test_ultimate_ml.py::TestUltimateML::test_initialization -v
```

### Using Docker
```bash
# Build and run tests
docker-compose up ara-test

# Run development environment
docker-compose up ara-dev

# Run production environment
docker-compose up ara-prod
```

### Using tox (if configured)
```bash
# Run tests on all Python versions
tox

# Run tests on specific Python version
tox -e py311
```

## Platform-Specific Notes

### Linux (Ubuntu)
- Uses `apt-get` for system dependencies
- Fastest CI/CD execution
- Full test coverage

### Windows
- Uses PowerShell/CMD
- Slightly slower due to Windows overhead
- Full compatibility testing

### macOS
- Uses Homebrew for dependencies
- Limited to Python 3.10+ (for efficiency)
- Apple Silicon (M1/M2) compatibility

## Secrets Required

### For Full CI/CD
- `GITHUB_TOKEN` - Automatically provided by GitHub
- `PYPI_API_TOKEN` - For PyPI publishing (optional)
- `CODECOV_TOKEN` - For Codecov integration (optional)

## Badge Status

Add these badges to your README:

```markdown
[![CI/CD](https://github.com/MeridianAlgo/AraAI/workflows/CI/CD%20Pipeline%20-%20Cross-Platform%20Tests/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions)
[![codecov](https://codecov.io/gh/MeridianAlgo/AraAI/branch/main/graph/badge.svg)](https://codecov.io/gh/MeridianAlgo/AraAI)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
```

## Troubleshooting

### Tests Failing on Specific Platform
1. Check platform-specific dependencies
2. Review error logs in GitHub Actions
3. Test locally using Docker with target platform

### Slow CI/CD Execution
1. Reduce matrix combinations
2. Use caching for dependencies
3. Parallelize independent jobs

### Coverage Issues
1. Ensure pytest-cov is installed
2. Check coverage configuration in pytest.ini
3. Verify Codecov token is set

## Contributing

When contributing:
1. Ensure all tests pass locally
2. Add tests for new features
3. Update documentation
4. Follow code style guidelines (Black, isort)
5. CI/CD will automatically run on PR

## Maintenance

### Weekly Tasks
- Review failed scheduled runs
- Update dependencies if needed
- Check security scan results

### Monthly Tasks
- Review and update Python version matrix
- Update GitHub Actions versions
- Review and optimize CI/CD performance

### Release Tasks
- Update version in `meridianalgo/__init__.py`
- Update CHANGELOG.md
- Create and push tag
- Verify release workflow completes
- Test installation from release
