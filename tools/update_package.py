#!/usr/bin/env python3
"""
Update script for MeridianAlgo Python package
Handles version bumping, building, and deployment
"""

import os
import sys
import subprocess
import re
from datetime import datetime
from pathlib import Path

def run_command(command, description, capture_output=True):
    """Run a command and handle errors"""
    print(f"\n {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=capture_output, 
            text=True
        )
        print(f" {description} completed successfully")
        if capture_output and result.stdout:
            print(result.stdout)
        return True, result.stdout if capture_output else ""
    except subprocess.CalledProcessError as e:
        print(f" {description} failed:")
        print(f"Error: {e}")
        if capture_output:
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
        return False, ""

def get_current_version():
    """Get current version from setup.py"""
    try:
        with open('setup.py', 'r') as f:
            content = f.read()
            version_match = re.search(r'version="([^"]+)"', content)
            if version_match:
                return version_match.group(1)
    except FileNotFoundError:
        pass
    
    # Try pyproject.toml
    try:
        with open('pyproject.toml', 'r') as f:
            content = f.read()
            version_match = re.search(r'version = "([^"]+)"', content)
            if version_match:
                return version_match.group(1)
    except FileNotFoundError:
        pass
    
    # Try __init__.py
    try:
        with open('meridianalgo/__init__.py', 'r') as f:
            content = f.read()
            version_match = re.search(r'__version__ = "([^"]+)"', content)
            if version_match:
                return version_match.group(1)
    except FileNotFoundError:
        pass
    
    return None

def parse_version(version_str):
    """Parse version string into components"""
    parts = version_str.split('.')
    if len(parts) >= 3:
        return int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]), int(parts[1]), 0
    else:
        return int(parts[0]), 0, 0

def increment_version(version_str, increment_type):
    """Increment version based on type"""
    major, minor, patch = parse_version(version_str)
    
    if increment_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif increment_type == 'minor':
        minor += 1
        patch = 0
    elif increment_type == 'patch':
        patch += 1
    
    return f"{major}.{minor}.{patch}"

def get_version_increment_choice():
    """Get user choice for version increment"""
    current_version = get_current_version()
    if not current_version:
        print(" Could not determine current version")
        return None, None
    
    print(f"\n Current version: {current_version}")
    print("\n Choose version increment type:")
    print(f"1. Patch (bug fixes): {current_version} → {increment_version(current_version, 'patch')}")
    print(f"2. Minor (new features): {current_version} → {increment_version(current_version, 'minor')}")
    print(f"3. Major (breaking changes): {current_version} → {increment_version(current_version, 'major')}")
    print("4. Custom version")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                return increment_version(current_version, 'patch'), 'patch'
            elif choice == '2':
                return increment_version(current_version, 'minor'), 'minor'
            elif choice == '3':
                return increment_version(current_version, 'major'), 'major'
            elif choice == '4':
                custom_version = input("Enter custom version (e.g., 2.1.0): ").strip()
                if re.match(r'^\d+\.\d+\.\d+$', custom_version):
                    return custom_version, 'custom'
                else:
                    print(" Invalid version format. Use X.Y.Z format.")
            else:
                print(" Invalid choice. Please enter 1-4.")
        except KeyboardInterrupt:
            print("\n Update cancelled by user")
            return None, None

def update_version_in_files(new_version):
    """Update version in all relevant files"""
    print(f"\n Updating version to {new_version} in all files...")
    
    files_to_update = [
        ('setup.py', r'version="[^"]+"', f'version="{new_version}"'),
        ('pyproject.toml', r'version = "[^"]+"', f'version = "{new_version}"'),
        ('meridianalgo/__init__.py', r'__version__ = "[^"]+"', f'__version__ = "{new_version}"'),
    ]
    
    updated_files = []
    
    for file_path, pattern, replacement in files_to_update:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                new_content = re.sub(pattern, replacement, content)
                
                if new_content != content:
                    with open(file_path, 'w') as f:
                        f.write(new_content)
                    updated_files.append(file_path)
                    print(f"   Updated {file_path}")
                else:
                    print(f"   No version found in {file_path}")
                    
            except Exception as e:
                print(f"   Failed to update {file_path}: {e}")
                return False
    
    if updated_files:
        print(f" Updated version in {len(updated_files)} files")
        return True
    else:
        print(" No files were updated")
        return False

def update_changelog(new_version, increment_type):
    """Update CHANGELOG.md with new version"""
    print(f"\n Updating CHANGELOG.md...")
    
    changelog_path = 'CHANGELOG.md'
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Create changelog entry
    if increment_type == 'major':
        change_type = "Major Release"
    elif increment_type == 'minor':
        change_type = "Feature Release"
    elif increment_type == 'patch':
        change_type = "Bug Fix Release"
    else:
        change_type = "Release"
    
    new_entry = f"""
## [{new_version}] - {today}

### {change_type}
- Enhanced Ara AI integration with latest improvements
- Updated ensemble ML models with better accuracy
- Improved GPU support and performance optimizations
- Enhanced caching system and prediction validation
- Updated documentation and examples

"""
    
    try:
        if os.path.exists(changelog_path):
            with open(changelog_path, 'r') as f:
                content = f.read()
            
            # Insert new entry after the header
            lines = content.split('\n')
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('## [') and 'Unreleased' not in line:
                    header_end = i
                    break
                elif line.startswith('# ') and i > 0:
                    header_end = i + 1
                    break
            
            if header_end > 0:
                lines.insert(header_end, new_entry.strip())
                new_content = '\n'.join(lines)
            else:
                new_content = content + new_entry
        else:
            # Create new changelog
            new_content = f"""# Changelog

All notable changes to MeridianAlgo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
{new_entry}
"""
        
        with open(changelog_path, 'w') as f:
            f.write(new_content)
        
        print(f" Updated {changelog_path}")
        return True
        
    except Exception as e:
        print(f" Failed to update changelog: {e}")
        return False

def commit_changes(new_version):
    """Commit version changes to git"""
    print(f"\n Committing changes to git...")
    
    # Check if git is available and we're in a git repo
    success, _ = run_command("git status", "Checking git status")
    if not success:
        print(" Not in a git repository or git not available. Skipping git operations.")
        return True
    
    # Add changed files
    files_to_add = [
        'setup.py',
        'pyproject.toml', 
        'meridianalgo/__init__.py',
        'CHANGELOG.md'
    ]
    
    for file_path in files_to_add:
        if os.path.exists(file_path):
            run_command(f"git add {file_path}", f"Adding {file_path}")
    
    # Commit changes
    commit_message = f"Bump version to {new_version}"
    success, _ = run_command(f'git commit -m "{commit_message}"', "Committing changes")
    
    if success:
        # Create git tag
        tag_name = f"v{new_version}"
        run_command(f'git tag {tag_name}', f"Creating tag {tag_name}")
        
        print(f" Created git commit and tag for version {new_version}")
        print(f" Don't forget to push: git push origin main --tags")
    
    return success

def build_and_deploy():
    """Build and deploy the package"""
    print(f"\n Building and deploying package...")
    
    # Run build script
    if os.path.exists('build_package.py'):
        success, _ = run_command("python build_package.py", "Building package", capture_output=False)
        if not success:
            return False
    else:
        print(" build_package.py not found")
        return False
    
    # Ask if user wants to deploy
    deploy_choice = input("\n Deploy to PyPI? (yes/no): ").strip().lower()
    if deploy_choice == 'yes':
        if os.path.exists('deploy_package.py'):
            success, _ = run_command("python deploy_package.py", "Deploying package", capture_output=False)
            return success
        else:
            print(" deploy_package.py not found")
            return False
    else:
        print(" Package built successfully. Deployment skipped.")
        return True

def main():
    """Main update process"""
    print(" MeridianAlgo Package Update")
    print("=" * 50)
    
    # Step 1: Get version increment choice
    new_version, increment_type = get_version_increment_choice()
    if not new_version:
        return 1
    
    print(f"\n Updating to version {new_version}")
    
    # Step 2: Update version in files
    if not update_version_in_files(new_version):
        print("\n Update failed: Could not update version in files")
        return 1
    
    # Step 3: Update changelog
    if not update_changelog(new_version, increment_type):
        print("\n Update failed: Could not update changelog")
        return 1
    
    # Step 4: Commit changes
    if not commit_changes(new_version):
        print("\n Git operations failed, but continuing...")
    
    # Step 5: Build and deploy
    if not build_and_deploy():
        print("\n Update failed: Build/deployment error")
        return 1
    
    print(f"\n Package update to version {new_version} completed successfully!")
    print("\n Next steps:")
    print("  1. Push git changes: git push origin main --tags")
    print("  2. Verify package on PyPI: https://pypi.org/project/meridianalgo/")
    print("  3. Test installation: pip install --upgrade meridianalgo")
    print("  4. Update documentation if needed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())