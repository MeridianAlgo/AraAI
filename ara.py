#!/usr/bin/env python3
"""
Ara AI Stock Analysis - Direct CLI Entrypoint
Allows: python ara.py <SYMBOL> [OPTIONS]
"""
import sys
from meridianalgo.cli import main

if __name__ == "__main__":
    sys.exit(main())