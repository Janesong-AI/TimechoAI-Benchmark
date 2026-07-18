"""
utils - Infrastructure Utility Layer

This package provides stateless pure functions and generic entity wrappers, serving as the 
underlying support for the entire project. It currently includes:

Modules:
  client.py — TimechoAI client connection entity.
    Provides factory functions get_timecho_client() / get_timecho_async_client(),
    unifying the creation and lifecycle management of TimechoAIClient / TimechoAIAsyncClient instances.
    Prevents duplicate handling of API_KEY and initialization logic across modules.
    Also provides reset_client() / reset_async_client() for resetting singletons in test scenarios.
  files.py — File operation utilities.
    Provides functionality for reading, writing, appending, and status checking for files such as CSV/JSON.

Usage Conventions:
  Business modules (e.g., features/) should access TimechoAI services indirectly through core.timecho.
  The core layer is the only module that directly uses utils.client.
  As a general utility, files can be used directly by any module.

  Import Path Examples:
    from core.timecho import forecast  # Recommended way
    # Instead of:
    # from utils.client import get_timecho_client  # Business modules should avoid this
"""

from .client import (
    get_timecho_client,
    get_timecho_async_client,
    reset_client,
    reset_async_client,
)
from .files import (
    save_to_csv,
    append_to_csv,
    save_with_json_backup,
    read_csv_to_dataframe,
    read_csv_to_list,
    csv_exists_and_not_empty,
    CSVFileError
)

__all__ = [
    # --- client.py ---
    "get_timecho_client",
    "get_timecho_async_client",
    "reset_client",
    "reset_async_client",
    # --- files.py ---
    "save_to_csv",
    "append_to_csv",
    "save_with_json_backup",
    "read_csv_to_dataframe",
    "read_csv_to_list",
    "csv_exists_and_not_empty",
    "CSVFileError",
]
