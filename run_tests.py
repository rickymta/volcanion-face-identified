"""
Comprehensive test runner and configuration
"""
import pytest
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_all_tests():
    """Run all tests with comprehensive reporting"""
    print("🧪 Running comprehensive test suite...")
    
    # Test discovery patterns
    test_patterns = [
        "tests/test_*.py",
        "tests/*/test_*.py",
        "domain/*/test_*.py",
        "application/*/test_*.py",
        "infrastructure/*/test_*.py",
        "presentation/*/test_*.py"
    ]
    
    # Pytest configuration
    pytest_args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker validation
        "--disable-warnings",  # Disable warnings for cleaner output
        "--color=yes",  # Colored output
        "-x",  # Stop on first failure (remove for full test run)
        "--junit-xml=test-results.xml",  # JUnit XML output
        "--cov=domain",  # Coverage for domain layer
        "--cov=application",  # Coverage for application layer  
        "--cov=infrastructure",  # Coverage for infrastructure layer
        "--cov=presentation",  # Coverage for presentation layer
        "--cov-report=html:htmlcov",  # HTML coverage report
        "--cov-report=term-missing",  # Terminal coverage with missing lines
        "--cov-fail-under=80",  # Minimum coverage threshold
    ]
    
    # Add test patterns
    for pattern in test_patterns:
        if Path(pattern.replace("test_*.py", "")).exists():
            pytest_args.append(pattern)
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code: {exit_code}")
    
    return exit_code

def run_specific_tests(test_path: str):
    """Run specific test file or directory"""
    print(f"🧪 Running tests: {test_path}")
    
    pytest_args = [
        "-v",
        "--tb=short", 
        "--color=yes",
        test_path
    ]
    
    return pytest.main(pytest_args)

def run_monitoring_tests():
    """Run only monitoring-related tests"""
    print("📊 Running monitoring tests...")
    
    pytest_args = [
        "-v",
        "--tb=short",
        "--color=yes",
        "tests/test_monitoring.py",
        "-k", "monitoring"
    ]
    
    return pytest.main(pytest_args)

def run_api_tests():
    """Run only API integration tests"""
    print("🌐 Running API tests...")
    
    pytest_args = [
        "-v", 
        "--tb=short",
        "--color=yes",
        "tests/",
        "-k", "api"
    ]
    
    return pytest.main(pytest_args)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test runner for Face Verification System")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--monitoring", action="store_true", help="Run monitoring tests")
    parser.add_argument("--api", action="store_true", help="Run API tests")
    parser.add_argument("--path", type=str, help="Run specific test path")
    
    args = parser.parse_args()
    
    if args.all:
        exit_code = run_all_tests()
    elif args.monitoring:
        exit_code = run_monitoring_tests()
    elif args.api:
        exit_code = run_api_tests()
    elif args.path:
        exit_code = run_specific_tests(args.path)
    else:
        print("Please specify test type. Use --help for options.")
        exit_code = 1
    
    sys.exit(exit_code)
