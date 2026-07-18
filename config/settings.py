"""
config/settings.py -- Global Configuration

Centrally manages all environment-configurable items.
This module is the single source of truth for paths and secrets.
"""

import os
from pathlib import Path

# ============================================================
# Path Constants
# ============================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Derived paths (defined centrally here; other modules use read-only references)
DATA_DIR:    Path = PROJECT_ROOT / "data"
OUTPUT_DIR:  Path = PROJECT_ROOT / "outputs"

# ============================================================
# API Configuration
# ============================================================
# Priority: Environment variables > Default values here
# If modification is needed, it is recommended to set TIMECHO_API_KEY in .env or via environment variables.

API_KEY: str = os.getenv(
    "TIMECHO_API_KEY",
    "ts-Update-Your-TIMECHO_API_KEY",
)

# Automatically ensure key directories exist (idempotent operation)
for _dir in (DATA_DIR, OUTPUT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

