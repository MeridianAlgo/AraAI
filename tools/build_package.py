#!/usr/bin/env python3
"""
Build script for MeridianAlgo Python package
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def clean_build_directories():
    """Clean previous build artifacts"""
    print("\n🧹 Cleaning build directories...")
    
    directories_to_clean = ['build', 'dist', 'meridianalgo.egg-info']
    
    for directory in directories_to_clean:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"  Removed: {directory}")
    
    # Clean __pycache__ directories
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                pycache_path = os.path.join(root, dir_name)
                shutil.rmtree(pycache_path)
                print(f"  Removed: {pycache_path}")

def validate_package_structure():
    """Validate package structure"""
    print("\n🔍 Validating package structure...")
    
    required_files = [
        'setup.py',
        'pyproject.toml',
        'README.md',
        'requirements.txt',
        'meridianalgo/__init__.py',
        'meridianalgo/core.py',
        'meridianalgo/models.py',
        'meridianalgo/data.py',
        'meridianalgo/utils.py',
        'meridianalgo/console.py',
        'meridianalgo/cli.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✅ Package structure is valid")
    return True

def install_build_dependencies():
    """Install build dependencies"""
    print("\n📦 Installing build dependencies...")
    
    build_deps = [
        'setuptools>=45',
        'wheel',
        'twine',
        'build'
    ]
    
    for dep in build_deps:
        if not run_command(f"python -m pip install {dep}", f"Installing {dep}"):
            return False
    
    return True

def build_package():
    """Build the package"""
    print("\n🏗️ Building package...")
    
    # Build source distribution
    if not run_command("python -m build --sdist", "Building source distribution"):
        return False
    
    # Build wheel distribution
    if not run_command("python -m build --wheel", "Building wheel distribution"):
        return False
    
    return True

def validate_built_package():
    """Validate the built package"""
    print("\n✅ Validating built package...")
    
    # Check if dist directory exists and has files
    if not os.path.exists('dist'):
        print("❌ dist directory not found")
        return False
    
    dist_files = os.listdir('dist')
    if not dist_files:
        print("❌ No files in dist directory")
        return False
    
    print(f"📦 Built files:")
    for file_name in dist_files:
        file_path = os.path.join('dist', file_name)
        file_size = os.path.getsize(file_path)
        print(f"  - {file_name} ({file_size:,} bytes)")
    
    # Validate with twine
    if not run_command("twine check dist/*", "Validating package with twine"):
        return False
    
    return True

def test_installation():
    """Test package installation"""
    print("\n🧪 Testing package installation...")
    
    # Create a temporary virtual environment for testing
    test_env = "test_env"
    
    # Clean up any existing test environment
    if os.path.exists(test_env):
        shutil.rmtree(test_env)
    
    # Create virtual environment
    if not run_command(f"python -m venv {test_env}", "Creating test environment"):
        return False
    
    # Determine activation script based on OS
    if sys.platform == "win32":
        activate_script = f"{test_env}\\Scripts\\activate"
        pip_command = f"{test_env}\\Scripts\\pip"
        python_command = f"{test_env}\\Scripts\\python"
    else:
        activate_script = f"{test_env}/bin/activate"
        pip_command = f"{test_env}/bin/pip"
        python_command = f"{test_env}/bin/python"
    
    # Install the package from dist
    wheel_files = [f for f in os.listdir('dist') if f.endswith('.whl')]
    if not wheel_files:
        print("❌ No wheel file found for testing")
        return False
    
    wheel_file = os.path.join('dist', wheel_files[0])
    
    if not run_command(f"python -m pip install {wheel_file}", "Installing package in test environment"):
        shutil.rmtree(test_env)
        return False
    
    # Test import
    test_script = """
import meridianalgo
print(f"MeridianAlgo version: {meridianalgo.__version__}")
print("Package import successful")

# Test basic functionality
from meridianalgo.utils import GPUManager
gpu_manager = GPUManager()
print(f"GPU info: {gpu_manager.detect_gpu_vendor()}")
print("Basic functionality test passed")
"""
    
    with open('test_import.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    success = run_command(f"{python_command} test_import.py", "Testing package import")
    
    # Clean up
    os.remove('test_import.py')
    shutil.rmtree(test_env)
    
    return success

def main():
    """Main build process"""
    print("🚀 MeridianAlgo Package Build Process")
    print("=" * 50)
    
    # Step 1: Clean build directories
    clean_build_directories()
    
    # Step 2: Validate package structure
    if not validate_package_structure():
        print("\n❌ Build failed: Invalid package structure")
        return 1
    
    # Step 3: Install build dependencies
    if not install_build_dependencies():
        print("\n❌ Build failed: Could not install build dependencies")
        return 1
    
    # Step 4: Build package
    if not build_package():
        print("\n❌ Build failed: Package build error")
        return 1
    
    # Step 5: Validate built package
    if not validate_built_package():
        print("\n❌ Build failed: Package validation error")
        return 1
    
    # Step 6: Test installation
    if not test_installation():
        print("\n❌ Build failed: Installation test error")
        return 1
    
    print("\n🎉 Package build completed successfully!")
    print("\n📦 Ready for deployment:")
    print("  - Source distribution: dist/*.tar.gz")
    print("  - Wheel distribution: dist/*.whl")
    print("\n🚀 To upload to PyPI:")
    print("  twine upload dist/*")
    print("\n🧪 To upload to Test PyPI:")
    print("  twine upload --repository testpypi dist/*")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())