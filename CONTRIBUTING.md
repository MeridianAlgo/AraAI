# Contributing to ARA AI

Thank you for your interest in contributing to ARA AI! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We pledge to:

- Be respectful and considerate
- Welcome diverse perspectives and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on collaboration and mutual benefit
- Show empathy and kindness to others

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or insults
- Trolling or inflammatory comments
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

**Required Information:**
- Clear and descriptive title
- Step-by-step reproduction instructions
- Expected behavior vs actual behavior
- Code samples or error messages
- System information (OS, Python version, etc.)
- ARA AI version

**Bug Report Template:**

```markdown
**Description**
A clear description of the bug.

**Steps to Reproduce**
1. Step one
2. Step two
3. ...

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python Version: [e.g., 3.11.5]
- ARA AI Version: [e.g., 4.0.0]

**Additional Context**
Any other relevant information.

**Error Messages/Logs**
```
Paste error messages or logs here
```
```

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- Clear and descriptive title
- Detailed description of the proposed feature
- Use cases and benefits
- Possible implementation approach
- Alternative solutions considered

**Enhancement Template:**

```markdown
**Feature Description**
A clear description of the feature.

**Problem It Solves**
What problem does this feature solve?

**Proposed Solution**
How should this feature work?

**Use Cases**
When would users use this feature?

**Alternatives Considered**
What other solutions did you consider?

**Additional Context**
Mockups, examples, or other context.
```

### Pull Requests

We welcome pull requests! Follow these steps:

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/AraAI.git
   cd AraAI
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
   
   Branch naming conventions:
   - `feature/feature-name` - New features
   - `bugfix/bug-description` - Bug fixes
   - `docs/documentation-update` - Documentation updates
   - `refactor/refactor-description` - Code refactoring
   - `test/test-description` - Test additions/updates

3. **Make Your Changes**
   - Follow the coding style guidelines
   - Write clear commit messages
   - Add tests for new features
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run specific test files
   pytest tests/test_your_feature.py
   
   # Check code quality
   black ara/ meridianalgo/
   flake8 ara/ meridianalgo/
   mypy ara/ meridianalgo/
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add feature: description of feature"
   ```
   
   Commit message guidelines:
   - Use present tense ("Add feature" not "Added feature")
   - Use imperative mood ("Move cursor to..." not "Moves cursor to...")
   - Limit first line to 72 characters
   - Reference issues and pull requests when relevant

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Submit a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template
   - Link related issues

### Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Code follows the project style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts
- [ ] PR description clearly explains changes
- [ ] Related issues are linked

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- pip or conda

### Setting Up Development Environment

1. **Clone the Repository**
   ```bash
   git clone https://github.com/meridianalgo/AraAI.git
   cd AraAI
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   # Install main dependencies
   pip install -r requirements.txt
   
   # Install development dependencies
   pip install pytest black flake8 mypy bandit safety
   ```

4. **Set Up Pre-commit Hooks** (Optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ara --cov=meridianalgo --cov-report=html

# Run specific test file
pytest tests/test_specific.py -v

# Run specific test
pytest tests/test_specific.py::test_function_name -v
```

### Code Quality Checks

```bash
# Format code with Black
black ara/ meridianalgo/

# Check code style with flake8
flake8 ara/ meridianalgo/ --max-line-length=100

# Type checking with mypy
mypy ara/ meridianalgo/

# Security scan with Bandit
bandit -r ara/ meridianalgo/

# Dependency vulnerability check
safety check
```

## Coding Style Guidelines

### Python Style

We follow PEP 8 with some modifications:

- **Line Length**: 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings
- **Imports**: Organized in three groups (standard library, third-party, local)

### Code Formatting

Use Black for automatic formatting:

```bash
black ara/ meridianalgo/
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `UnifiedStockML`)
- **Functions**: `snake_case` (e.g., `calculate_indicators`)
- **Variables**: `snake_case` (e.g., `prediction_result`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_PREDICTIONS`)
- **Private**: Prefix with `_` (e.g., `_internal_function`)

### Documentation

- Use docstrings for all public functions, classes, and modules
- Follow Google-style docstrings
- Include type hints for function parameters and return values

**Example:**

```python
def calculate_prediction(
    symbol: str,
    days: int,
    model_type: str = "ensemble"
) -> Dict[str, Any]:
    """
    Calculate price prediction for a given symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        days: Number of days to predict
        model_type: Type of model to use
        
    Returns:
        Dictionary containing prediction results with keys:
            - predictions: List of daily predictions
            - confidence: Overall confidence score
            - metadata: Additional prediction metadata
            
    Raises:
        ValueError: If symbol is invalid or days is out of range
        
    Example:
        >>> result = calculate_prediction('AAPL', days=5)
        >>> print(result['predictions'])
    """
    # Implementation
    pass
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import List, Dict, Optional, Union, Any

def process_data(
    data: List[float],
    config: Optional[Dict[str, Any]] = None
) -> Union[List[float], None]:
    # Implementation
    pass
```

### Error Handling

- Use specific exceptions rather than generic ones
- Provide helpful error messages
- Log errors appropriately
- Don't catch exceptions unless you can handle them

```python
from ara.exceptions import InvalidSymbolError

def validate_symbol(symbol: str) -> str:
    if not symbol or not symbol.isalpha():
        raise InvalidSymbolError(
            f"Invalid symbol '{symbol}'. Symbol must contain only letters."
        )
    return symbol.upper()
```

## Testing Guidelines

### Test Structure

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Group related tests in classes

### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Test both success and failure paths
- Mock external dependencies

### Example Test

```python
import pytest
from ara.features.calculator import IndicatorCalculator

class TestIndicatorCalculator:
    """Tests for IndicatorCalculator class."""
    
    def test_calculate_rsi_valid_input(self):
        """Test RSI calculation with valid data."""
        calc = IndicatorCalculator()
        data = [100, 102, 101, 103, 105]
        
        result = calc.calculate_rsi(data, period=3)
        
        assert result is not None
        assert isinstance(result, float)
        assert 0 <= result <= 100
    
    def test_calculate_rsi_insufficient_data(self):
        """Test RSI calculation with insufficient data."""
        calc = IndicatorCalculator()
        data = [100, 102]  # Too few points
        
        with pytest.raises(ValueError):
            calc.calculate_rsi(data, period=3)
```

## Documentation Guidelines

### Code Documentation

- Add docstrings to all public APIs
- Include examples in docstrings
- Document complex algorithms
- Add inline comments for non-obvious code

### README Updates

When adding new features:
- Update feature list
- Add usage examples
- Update installation instructions if needed
- Add to table of contents

### Creating New Documentation

For major features:
- Create a new markdown file in the relevant directory
- Add comprehensive usage examples
- Include troubleshooting section
- Link from main README

## Project Structure

```
AraAI/
├── ara/                    # Main package
│   ├── api/               # REST API
│   ├── models/            # ML models
│   ├── data/              # Data providers
│   ├── features/          # Feature engineering
│   ├── risk/              # Risk management
│   ├── backtesting/       # Backtesting engine
│   ├── sentiment/         # Sentiment analysis
│   ├── security/          # Security features
│   └── monitoring/        # Monitoring and metrics
├── meridianalgo/          # Core ML algorithms
├── scripts/               # Utility scripts
├── tests/                 # Test suite
├── docs/                  # Documentation
├── datasets/              # Sample datasets
├── models/                # Trained models
├── requirements.txt       # Dependencies
├── LICENSE               # License file
├── README.md             # Main documentation
├── CONTRIBUTING.md       # This file
└── SECURITY.md           # Security policy
```

## Git Workflow

### Branching Strategy

- `main` - Stable production code
- `develop` - Integration branch for features
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Critical production fixes

### Commit Guidelines

**Good commit messages:**
```
Add RSI indicator calculation

- Implement RSI calculation using pandas
- Add tests for RSI calculator
- Update documentation with usage examples

Fixes #123
```

**Bad commit messages:**
```
fixed stuff
update
changes
```

## Community

### Getting Help

- GitHub Issues - Bug reports and feature requests
- GitHub Discussions - Questions and community discussions
- Documentation - Comprehensive guides and API reference

### Communication Channels

- GitHub Issues - Technical discussions
- GitHub Discussions - General questions and ideas
- Pull Requests - Code reviews and collaboration

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

## License

By contributing to ARA AI, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search closed issues
3. Ask in GitHub Discussions
4. Open a new issue with the "question" label

---

Thank you for contributing to ARA AI! Your contributions help make financial prediction tools more accessible and powerful for everyone.

**Last Updated**: 2025-11-25
