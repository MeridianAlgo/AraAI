#!/usr/bin/env python3
"""
Deployment script for MeridianAlgo Python package to PyPI
"""

import os
import sys
import subprocess
import getpass
from pathlib import Path

def run_command(command, description, capture_output=True):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=capture_output, 
            text=True
        )
        print(f"✅ {description} completed successfully")
        if capture_output and result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e}")
        if capture_output:
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
        return False

def check_prerequisites():
    """Check deployment prerequisites"""
    print("\n🔍 Checking deployment prerequisites...")
    
    # Check if dist directory exists
    if not os.path.exists('dist'):
        print("❌ dist directory not found. Run build_package.py first.")
        return False
    
    # Check if there are files to upload
    dist_files = [f for f in os.listdir('dist') if f.endswith(('.whl', '.tar.gz'))]
    if not dist_files:
        print("❌ No distribution files found in dist/")
        return False
    
    print(f"📦 Found {len(dist_files)} distribution files:")
    for file_name in dist_files:
        file_path = os.path.join('dist', file_name)
        file_size = os.path.getsize(file_path)
        print(f"  - {file_name} ({file_size:,} bytes)")
    
    # Check if twine is installed
    try:
        subprocess.run(['twine', '--version'], check=True, capture_output=True)
        print("✅ twine is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ twine not found. Installing...")
        if not run_command("python -m pip install twine", "Installing twine"):
            return False
    
    return True

def validate_package():
    """Validate package before upload"""
    print("\n✅ Validating package...")
    
    return run_command("twine check dist/*", "Validating package with twine")

def get_deployment_choice():
    """Get user choice for deployment target"""
    print("\n🎯 Choose deployment target:")
    print("1. Test PyPI (recommended for testing)")
    print("2. Production PyPI (live deployment)")
    print("3. Both (Test PyPI first, then Production)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            if choice in ['1', '2', '3']:
                return int(choice)
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n⚠️ Deployment cancelled by user")
            return None

def deploy_to_test_pypi():
    """Deploy to Test PyPI"""
    print("\n🧪 Deploying to Test PyPI...")
    
    # Test PyPI repository URL
    test_pypi_url = "https://test.pypi.org/legacy/"
    
    command = f"twine upload --repository-url {test_pypi_url} dist/*"
    
    print("📝 You will be prompted for your Test PyPI credentials")
    print("   If you don't have an account, create one at: https://test.pypi.org/account/register/")
    
    return run_command(command, "Uploading to Test PyPI", capture_output=False)

def deploy_to_production_pypi():
    """Deploy to Production PyPI"""
    print("\n🚀 Deploying to Production PyPI...")
    
    print("⚠️  WARNING: This will deploy to the live PyPI repository!")
    print("   Make sure you have tested the package thoroughly.")
    
    confirm = input("\nAre you sure you want to deploy to Production PyPI? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("❌ Production deployment cancelled")
        return False
    
    command = "twine upload dist/*"
    
    print("📝 You will be prompted for your PyPI credentials")
    print("   If you don't have an account, create one at: https://pypi.org/account/register/")
    
    return run_command(command, "Uploading to Production PyPI", capture_output=False)

def verify_deployment(target):
    """Verify deployment was successful"""
    print(f"\n🔍 Verifying deployment to {target}...")
    
    if target == "Test PyPI":
        url = "https://test.pypi.org/project/meridianalgo/"
        install_command = "pip install --index-url https://test.pypi.org/simple/ meridianalgo"
    else:
        url = "https://pypi.org/project/meridianalgo/"
        install_command = "pip install meridianalgo"
    
    print(f"📦 Package should be available at: {url}")
    print(f"🔧 Install command: {install_command}")
    
    # Also show python -m pip version for Windows compatibility
    if "test.pypi.org" not in install_command:
        print(f"🔧 Alternative: python -m pip install meridianalgo")
    
    # Wait for user confirmation
    input("\nPress Enter after verifying the package is available online...")
    
    return True

def post_deployment_instructions():
    """Show post-deployment instructions"""
    print("\n🎉 Deployment completed successfully!")
    print("\n📋 Post-deployment checklist:")
    print("  ✅ Verify package is available on PyPI")
    print("  ✅ Test installation: pip install meridianalgo")
    print("  ✅ Test CLI: ara --version")
    print("  ✅ Update GitHub repository with new version tag")
    print("  ✅ Update documentation if needed")
    
    print("\n🔗 Useful links:")
    print("  - PyPI Package: https://pypi.org/project/meridianalgo/")
    print("  - Test PyPI: https://test.pypi.org/project/meridianalgo/")
    print("  - GitHub Repo: https://github.com/MeridianAlgo/Ara")
    
    print("\n📊 Package statistics:")
    print("  - Check download stats at: https://pypistats.org/packages/meridianalgo")

def main():
    """Main deployment process"""
    print("🚀 MeridianAlgo Package Deployment")
    print("=" * 50)
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Deployment failed: Prerequisites not met")
        return 1
    
    # Step 2: Validate package
    if not validate_package():
        print("\n❌ Deployment failed: Package validation error")
        return 1
    
    # Step 3: Get deployment choice
    choice = get_deployment_choice()
    if choice is None:
        return 1
    
    # Step 4: Deploy based on choice
    success = True
    
    if choice == 1:  # Test PyPI only
        success = deploy_to_test_pypi()
        if success:
            verify_deployment("Test PyPI")
    
    elif choice == 2:  # Production PyPI only
        success = deploy_to_production_pypi()
        if success:
            verify_deployment("Production PyPI")
    
    elif choice == 3:  # Both
        # Deploy to Test PyPI first
        if deploy_to_test_pypi():
            verify_deployment("Test PyPI")
            
            # Ask if user wants to continue to production
            continue_prod = input("\nDeploy to Production PyPI? (yes/no): ").strip().lower()
            if continue_prod == 'yes':
                success = deploy_to_production_pypi()
                if success:
                    verify_deployment("Production PyPI")
            else:
                print("✅ Test PyPI deployment completed. Production deployment skipped.")
        else:
            success = False
    
    if success:
        post_deployment_instructions()
        print("\n🎉 All deployments completed successfully!")
        return 0
    else:
        print("\n❌ Deployment failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())