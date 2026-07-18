#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TSFM-Robustness-Benchmark — Unified Entry Point

Usage:
  python run.py features.futureCovs.conceptDrift.test
  python run.py ./features/futureCovs/conceptDrift/test.py
"""
import os
import sys
import importlib
from pathlib import Path

# Bootstrap: Used solely for sys.path setup, not exposed as global variables.
_BOOTSTRAP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from config.settings import PROJECT_ROOT

# Run the specified module
if len(sys.argv) < 2:
    print("Usage: python run.py <module_name>")
    print("Example: python run.py features.futureCovs.conceptDrift.test")
    print("         python run.py ./features/futureCovs/conceptDrift/test.py")
    sys.exit(1)

raw_path = sys.argv[1]

# Auto-detect path format
if raw_path.startswith("./") or raw_path.endswith(".py") or "/" in raw_path:
    # File path format: ./features/xxx/test.py
    file_path = Path(raw_path).resolve()
    if not file_path.exists():
        print(f"Error: File does not exist: {file_path}")
        sys.exit(1)
    
    # Convert to module path
    # TSFM-Robustness-Benchmark/features/futureCovs/conceptDrift/test.py
    # → features.futureCovs.conceptDrift.test
    rel_path = file_path.relative_to(PROJECT_ROOT)
    # Remove .py extension
    if rel_path.suffix == ".py":
        rel_path = rel_path.with_suffix("")
    # Convert to dot-separated
    module_path = str(rel_path).replace("/", ".")
    # Compatible with Windows
    module_path = module_path.replace(os.sep, ".")
else:
    # Module path format: features.futureCovs.conceptDrift.test
    module_path = raw_path

print(f"Running module: {module_path}")
print(f"Project root:   {PROJECT_ROOT}")
print("-" * 50)

# Import and execute module
try:
    importlib.import_module(module_path)
except ModuleNotFoundError as e:
    print(f"\nError: Module {module_path} not found")
    print(f"Details: {e}")
    print("\nPossible reasons:")
    print("  1. Module path is incorrect")
    print("  2. Missing __init__.py file")
    sys.exit(1)