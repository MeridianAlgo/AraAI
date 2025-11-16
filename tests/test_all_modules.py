#!/usr/bin/env python3
"""
Comprehensive test runner for all ARA modules.
Tests everything except the tests folder itself.
"""
import sys
import importlib
import traceback
from pathlib import Path
from typing import List, Dict, Tuple

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class ModuleTester:
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'skipped': []
        }
    
    def test_import(self, module_path: str) -> Tuple[bool, str]:
        """Test if a module can be imported."""
        try:
            importlib.import_module(module_path)
            return True, "OK"
        except Exception as e:
            return False, str(e)
    
    def discover_modules(self, base_path: Path, package_prefix: str = "") -> List[str]:
        """Discover all Python modules in a directory."""
        modules = []
        
        for item in base_path.iterdir():
            if item.name.startswith('_') or item.name.startswith('.'):
                continue
                
            if item.is_file() and item.suffix == '.py':
                module_name = item.stem
                if package_prefix:
                    full_name = f"{package_prefix}.{module_name}"
                else:
                    full_name = module_name
                modules.append(full_name)
            
            elif item.is_dir() and (item / '__init__.py').exists():
                subpackage = f"{package_prefix}.{item.name}" if package_prefix else item.name
                modules.append(subpackage)
                modules.extend(self.discover_modules(item, subpackage))
        
        return modules
    
    def run_tests(self):
        """Run tests on all discovered modules."""
        print(f"{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}ARA Comprehensive Module Testing{RESET}")
        print(f"{BLUE}{'='*80}{RESET}\n")
        
        # Test main packages
        packages_to_test = ['ara', 'meridianalgo']
        
        for package in packages_to_test:
            package_path = Path(package)
            if not package_path.exists():
                print(f"{YELLOW}⚠ Package {package} not found, skipping{RESET}")
                continue
            
            print(f"\n{BLUE}Testing package: {package}{RESET}")
            print(f"{'-'*80}")
            
            modules = self.discover_modules(package_path, package)
            
            for module in sorted(modules):
                success, message = self.test_import(module)
                
                if success:
                    print(f"{GREEN}✓{RESET} {module}")
                    self.results['passed'].append(module)
                else:
                    print(f"{RED}✗{RESET} {module}")
                    print(f"  {RED}Error: {message[:100]}{RESET}")
                    self.results['failed'].append((module, message))
        
        # Test standalone scripts
        print(f"\n{BLUE}Testing standalone scripts{RESET}")
        print(f"{'-'*80}")
        
        scripts_dir = Path('scripts')
        if scripts_dir.exists():
            for script in scripts_dir.glob('*.py'):
                module_name = f"scripts.{script.stem}"
                # Just check if file is valid Python
                try:
                    with open(script, 'r', encoding='utf-8') as f:
                        compile(f.read(), script, 'exec')
                    print(f"{GREEN}✓{RESET} {script.name} (syntax valid)")
                    self.results['passed'].append(str(script))
                except SyntaxError as e:
                    print(f"{RED}✗{RESET} {script.name}")
                    print(f"  {RED}Syntax Error: {e}{RESET}")
                    self.results['failed'].append((str(script), str(e)))
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}Test Summary{RESET}")
        print(f"{BLUE}{'='*80}{RESET}")
        
        total = len(self.results['passed']) + len(self.results['failed']) + len(self.results['skipped'])
        passed = len(self.results['passed'])
        failed = len(self.results['failed'])
        skipped = len(self.results['skipped'])
        
        print(f"\nTotal modules tested: {total}")
        print(f"{GREEN}Passed: {passed}{RESET}")
        print(f"{RED}Failed: {failed}{RESET}")
        print(f"{YELLOW}Skipped: {skipped}{RESET}")
        
        if failed > 0:
            print(f"\n{RED}Failed modules:{RESET}")
            for module, error in self.results['failed']:
                print(f"  • {module}")
                print(f"    {error[:200]}")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"\n{BLUE}Success Rate: {success_rate:.1f}%{RESET}")
        
        if failed == 0:
            print(f"\n{GREEN}🎉 All modules passed!{RESET}")
            return 0
        else:
            print(f"\n{RED}❌ Some modules failed{RESET}")
            return 1

if __name__ == '__main__':
    tester = ModuleTester()
    exit_code = tester.run_tests()
    sys.exit(exit_code)
