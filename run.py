#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TSFM-Robustness-Benchmark unified entry point

Usage:
  python run.py features.futureCovs.conceptDrift.test
  python run.py ./features/futureCovs/conceptDrift/test.py
"""
import sys
import importlib
from pathlib import Path

# Set project path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    # TimechoAI-Benchmark/features/futureCovs/conceptDrift/test.py
    # → features.futureCovs.conceptDrift.test
    rel_path = file_path.relative_to(PROJECT_ROOT)
    # Remove .py extension
    if rel_path.suffix == ".py":
        rel_path = rel_path.with_suffix("")
    # Convert to dot-separated
    module_path = str(rel_path).replace("/", ".")
else:
    # Module path format: features.futureCovs.conceptDrift.test
    module_path = raw_path

print(f"Running module: {module_path}")
print(f"Project root: {PROJECT_ROOT}")
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
    print("\nCheck __init__.py files:")
    for dir_name in ["features", "futureCovs", "conceptDrift"]:
        init_file = PROJECT_ROOT / dir_name / "__init__.py"
        exists = "[OK]" if init_file.exists() else "[Missing]"
        print(f"  {dir_name}/__init__.py {exists}")
    sys.exit(1)