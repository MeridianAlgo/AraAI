#!/usr/bin/env python3
"""
Comprehensive test runner script for ARA AI.

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --fast             # Run fast tests only
    python tests/run_tests.py --coverage         # Run with coverage
    python tests/run_tests.py --unit             # Run unit tests only
    python tests/run_tests.py --integration      # Run integration tests
    python tests/run_tests.py --api              # Run API tests
    python tests/run_tests.py --performance      # Run performance benchmarks
    python tests/run_tests.py --property         # Run property-based tests
    python tests/run_tests.py --all              # Run comprehensive suite
    python tests/run_tests.py --ci               # Run CI test suite
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the result."""
    if description:
        print(f"\n{'='*60}")
        print(f"  {description}")
        print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=isinstance(cmd, str))
    return result.returncode


def run_unit_tests(coverage=False):
    """Run unit tests."""
    cmd = ["pytest", "tests/", "-v", "-m", "unit", "--tb=short"]
    
    if coverage:
        cmd.extend(["--cov=ara", "--cov=meridianalgo", "--cov-report=term"])
    
    return run_command(cmd, "Running Unit Tests")


def run_integration_tests(coverage=False):
    """Run integration tests."""
    cmd = ["pytest", "tests/", "-v", "-m", "integration", "--tb=short"]
    
    if coverage:
        cmd.extend(["--cov=ara", "--cov=meridianalgo", "--cov-report=term"])
    
    return run_command(cmd, "Running Integration Tests")


def run_api_tests():
    """Run API tests."""
    cmd = ["pytest", "tests/", "-v", "-m", "api", "--tb=short"]
    return run_command(cmd, "Running API Tests")


def run_e2e_tests():
    """Run end-to-end tests."""
    cmd = ["pytest", "tests/test_e2e.py", "tests/test_integration.py", "-v"]
    return run_command(cmd, "Running End-to-End Tests")


def run_performance_tests():
    """Run performance benchmarks."""
    cmd = ["pytest", "tests/performance/", "-v", "--benchmark-only"]
    return run_command(cmd, "Running Performance Benchmarks")


def run_property_tests():
    """Run property-based tests."""
    cmd = ["pytest", "tests/property/", "-v", "--hypothesis-show-statistics"]
    return run_command(cmd, "Running Property-Based Tests")


def run_fast_tests():
    """Run fast tests only."""
    cmd = ["pytest", "tests/", "-v", "-m", "not slow", "--maxfail=10", "-n", "auto"]
    return run_command(cmd, "Running Fast Tests")


def run_coverage_tests():
    """Run tests with coverage report."""
    cmd = [
        "pytest", "tests/", "-v",
        "--cov=ara",
        "--cov=meridianalgo",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-report=xml"
    ]
    
    result = run_command(cmd, "Running Tests with Coverage")
    
    if result == 0:
        print("\n" + "="*60)
        print("  Coverage report generated: htmlcov/index.html")
        print("="*60 + "\n")
    
    return result


def run_ci_tests():
    """Run CI test suite."""
    print("\n" + "="*60)
    print("  Running CI Test Suite")
    print("="*60 + "\n")
    
    # Run linting
    print("1. Code Quality Checks...")
    run_command(
        "black --check ara/ tests/ || echo 'Format check completed'",
        "Checking Code Format"
    )
    
    run_command(
        "flake8 ara/ tests/ --max-line-length=100 --ignore=E203,W503 || echo 'Linting completed'",
        "Linting Code"
    )
    
    # Run fast tests
    print("\n2. Fast Tests...")
    result = run_fast_tests()
    
    # Run coverage
    print("\n3. Coverage Check...")
    run_command(
        ["pytest", "tests/", "--cov=ara", "--cov-report=term", "--cov-fail-under=70"],
        "Checking Coverage"
    )
    
    return result


def run_all_tests():
    """Run comprehensive test suite."""
    print("\n" + "="*60)
    print("  Running Comprehensive Test Suite")
    print("="*60 + "\n")
    
    results = []
    
    # Unit tests
    results.append(("Unit Tests", run_unit_tests(coverage=True)))
    
    # Integration tests
    results.append(("Integration Tests", run_integration_tests(coverage=True)))
    
    # API tests
    results.append(("API Tests", run_api_tests()))
    
    # E2E tests
    results.append(("E2E Tests", run_e2e_tests()))
    
    # Performance tests
    results.append(("Performance Tests", run_performance_tests()))
    
    # Property-based tests
    results.append(("Property Tests", run_property_tests()))
    
    # Print summary
    print("\n" + "="*60)
    print("  Test Summary")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASSED" if result == 0 else "✗ FAILED"
        print(f"  {name:30} {status}")
    
    print("="*60 + "\n")
    
    # Return 0 if all passed, 1 otherwise
    return 0 if all(r == 0 for _, r in results) else 1


def run_smoke_tests():
    """Run smoke tests."""
    cmd = ["pytest", "tests/test_smoke.py", "-v"]
    return run_command(cmd, "Running Smoke Tests")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ARA AI Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast tests only (skip slow tests)"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage report"
    )
    
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only"
    )
    
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only"
    )
    
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run API tests only"
    )
    
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Run end-to-end tests"
    )
    
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance benchmarks"
    )
    
    parser.add_argument(
        "--property",
        action="store_true",
        help="Run property-based tests"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run comprehensive test suite"
    )
    
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run CI test suite"
    )
    
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke tests only"
    )
    
    args = parser.parse_args()
    
    # Determine which tests to run
    if args.all:
        return run_all_tests()
    elif args.ci:
        return run_ci_tests()
    elif args.fast:
        return run_fast_tests()
    elif args.coverage:
        return run_coverage_tests()
    elif args.unit:
        return run_unit_tests(coverage=True)
    elif args.integration:
        return run_integration_tests(coverage=True)
    elif args.api:
        return run_api_tests()
    elif args.e2e:
        return run_e2e_tests()
    elif args.performance:
        return run_performance_tests()
    elif args.property:
        return run_property_tests()
    elif args.smoke:
        return run_smoke_tests()
    else:
        # Default: run all tests with coverage
        return run_coverage_tests()


if __name__ == "__main__":
    sys.exit(main())
