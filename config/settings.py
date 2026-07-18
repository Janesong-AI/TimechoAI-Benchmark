"""
config/settings.py -- Global Configuration
Centrally manages all environment-configurable items, including API keys, etc.
"""

import os
from pathlib import Path

# ============================================================
# API Configuration
# ============================================================
# Priority: Environment variables > Default values here
# If modification is needed, it is recommended to set TIMECHO_API_KEY in .env or via environment variables.

API_KEY = os.getenv(
    "TIMECHO_API_KEY",
    "ts-Update-Your-TIMECHO_API_KEY",
)

