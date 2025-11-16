# CI/CD Workflows

Comprehensive CI/CD pipeline for ARA AI with automated testing, coverage reporting, and performance monitoring.

## Workflows

### 1. `ci.yml` - Main CI/CD Pipeline

**Triggers:**
- Push to main/develop branches
- Pull requests to main/develop
- Daily at 2 AM UTC (scheduled)

**Jobs:**

#### Lint and Format Check
- Black code formatting check
- Flake8 linting
- Runs on: Ubuntu Latest

#### Test Suite
- Runs on: Ubuntu, Windows, macOS
- Python versions: 3.9, 3.10, 3.11
- Unit tests
- Integration tests
- Smoke tests

#### Code Coverage
- Generates coverage reports (XML, HTML, term)
- Uploads to Codecov
- Archives HTML coverage report
- Target: 80%+ coverage

#### Performance Tests
- Runs performance benchmarks
- Tracks performance regressions
- Stores benchmark results

#### Backtest Validation
- Automated model accuracy validation
- Runs on schedule and push events
- Archives backtest results

#### Release (on version change)
- Auto-creates GitHub release
- Tags new versions
- Generates release notes

### 2. `nightly-tests.yml` - Comprehensive Nightly Tests

**Triggers:**
- Daily at 1 AM UTC
- Manual trigger (workflow_dispatch)

**Jobs:**

#### Comprehensive Tests
- All tests including slow tests
- Parallel execution with pytest-xdist
- Full coverage report
- 60-minute timeout

#### Property-Based Tests
- Hypothesis-based testing
- Generates random test cases
- Shows statistics

#### Stress and Load Tests
- API load testing with Locust
- Stress testing critical paths
- Performance under load

#### Model Accuracy Check
- Validates all models
- Checks accuracy thresholds
- Alerts on degradation

#### Notification
- Sends alerts on failure
- Can integrate with Slack/email

### 3. `pr-tests.yml` - Pull Request Tests

**Triggers:**
- Pull request opened/synchronized/reopened

**Jobs:**

#### Quick Tests
- Fast tests only (< 10 minutes)
- Parallel execution
- Coverage check (70% minimum)

#### Code Quality
- Black formatting check
- Flake8 linting
- MyPy type checking
- Pylint analysis

#### Security Scan
- Bandit security scanning
- Safety dependency check
- Uploads security reports

#### API Tests
- Integration tests for API endpoints
- Authentication tests
- WebSocket tests

#### Coverage Comment
- Posts coverage report to PR
- Shows coverage changes
- Highlights uncovered lines

## Running Workflows Locally

### Run CI Tests Locally
```bash
# Install dependencies
pip install pytest pytest-cov pytest-asyncio pytest-xdist

# Run CI test suite
python tests/run_tests.py --ci
```

### Run All Tests
```bash
python tests/run_tests.py --all
```

### Run Specific Test Categories
```bash
# Unit tests
python tests/run_tests.py --unit

# Integration tests
python tests/run_tests.py --integration

# API tests
python tests/run_tests.py --api

# Performance tests
python tests/run_tests.py --performance
```

### Check Coverage
```bash
python tests/run_tests.py --coverage
```

## Workflow Configuration

### Secrets Required
- `GITHUB_TOKEN` - Automatically provided by GitHub
- `CODECOV_TOKEN` - (Optional) For Codecov integration

### Environment Variables
None required for basic operation.

### Artifacts Generated
- Coverage reports (HTML)
- Benchmark results (JSON)
- Backtest results (JSON)
- Security scan reports (JSON)

## Release Process

### Automatic Release

1. Update version in `meridianalgo/__init__.py`:
```python
__version__ = "3.1.0"
```

2. Update version in `setup.py`:
```python
version='3.1.0',
```

3. Update version in `README.md`:
```markdown
# ARA AI v3.1.0
```

4. Commit and push:
```bash
git add -A
git commit -m "Release v3.1.0"
git push origin main
```

5. GitHub automatically:
   - Runs all tests
   - Creates tag `v3.1.0`
   - Creates GitHub release
   - Generates release notes

### Manual Release

Trigger release workflow manually from GitHub Actions UI.

## Monitoring and Alerts

### Test Failures
- CI fails on test failures
- PR cannot merge if tests fail
- Nightly tests send notifications

### Coverage Drops
- PR tests check coverage doesn't decrease
- Minimum 70% coverage required
- Full report posted to PR

### Performance Regressions
- Benchmarks tracked over time
- Alerts on significant slowdowns
- Performance dashboard available

### Security Issues
- Bandit scans for security issues
- Safety checks dependencies
- Reports uploaded as artifacts

## Best Practices

1. **Before Pushing**
   ```bash
   # Run fast tests
   python tests/run_tests.py --fast
   
   # Check formatting
   black ara/ tests/
   
   # Run linting
   flake8 ara/ tests/
   ```

2. **Before Creating PR**
   ```bash
   # Run full test suite
   python tests/run_tests.py --all
   
   # Check coverage
   python tests/run_tests.py --coverage
   ```

3. **For New Features**
   - Add unit tests
   - Add integration tests
   - Update documentation
   - Ensure coverage doesn't drop

4. **For Bug Fixes**
   - Add regression test
   - Verify fix with tests
   - Update changelog

## Troubleshooting

### Tests Failing in CI but Passing Locally
- Check Python version (CI uses 3.9, 3.10, 3.11)
- Check OS differences (CI tests on Ubuntu, Windows, macOS)
- Verify dependencies are installed
- Check for timing issues in async tests

### Coverage Report Not Generated
- Ensure pytest-cov is installed
- Check coverage configuration in pytest.ini
- Verify source paths are correct

### Workflow Not Triggering
- Check branch names (main/develop)
- Verify workflow file syntax
- Check GitHub Actions permissions

### Performance Tests Timing Out
- Increase timeout in workflow
- Optimize slow tests
- Use mocks for external dependencies

## Contributing

When modifying workflows:
1. Test changes in a fork first
2. Use workflow_dispatch for manual testing
3. Document any new secrets/variables
4. Update this README

**Maintained by**: ARA AI Team  
**Last Updated**: November 15, 2025
