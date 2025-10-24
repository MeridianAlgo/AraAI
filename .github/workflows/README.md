# CI/CD Workflows

## Simple CI/CD Pipeline

We use a single, simple workflow that:

1. **Tests** - Runs smoke tests on every push
2. **Auto Release** - Creates GitHub release when version changes

### Workflow: `ci.yml`

**Triggers:**
- Push to main branch
- Pull requests to main

**Jobs:**
1. **Test** - Runs smoke tests (5 min timeout)
2. **Release** - Auto-creates release if version changed

### How to Release

1. Update version in `meridianalgo/__init__.py`
2. Update version in `setup.py`
3. Update version in `README.md`
4. Commit and push to main
5. GitHub automatically creates release with new tag

### Example

```python
# meridianalgo/__init__.py
__version__ = "3.0.2"  # Change this

# setup.py
version='3.0.2',  # Change this

# README.md
# ARA AI - Ultimate Stock Prediction System v3.0.2  # Change this
[![Version](https://img.shields.io/badge/version-3.0.2-brightgreen.svg)]  # Change this
```

Then:
```bash
git add -A
git commit -m "Release v3.0.2"
git push origin main
```

GitHub will automatically:
- Run tests
- Create tag `v3.0.2`
- Create GitHub release

That's it! Simple and fast.
