"""
core -  Core Component Layer

This package encapsulates essential reusable components, serving as a bridge between 
the business layer and underlying utility services. It invokes services from the 
``utils`` layer and provides standard interfaces for business modules.

Modules:
  dataResults.py: Result Data Processing
    Contains utility functions for data cleaning and result retrieval.
  resume.py: Checkpoint & Resume Management
    Manages checkpoint status and file persistence, enabling recovery of long-running tasks after interruptions.
  timecho.py: TimechoAI API Interaction
    Handles API requests and response processing via ``utils.client``, offering a unified high-level API for external use.

Usage:
  Business modules (e.g., in ``features/``) should access TimechoAI services 
  via ``core.timecho``. The ``core`` layer should be the sole module directly 
  utilizing ``utils.client`` to ensure decoupling.

  Import Path Examples:
    # Recommended Approach
      from core.dataResults import clean_nan_values, get_results
      from core.timecho import forecast
      from core.resume import load_completed_results, append_result, is_rate_limited

    # Avoid direct imports from utils.client to prevent tight coupling
    # from utils.client import get_timecho_client
"""

from .dataResults import (
    clean_nan_values, 
    get_results
)

from .resume import (
    load_completed_results,
    append_result,
    is_rate_limited
)

from .timecho import (
    forecast,
    extract_pred_values,
    calc_metrics,
    calc_diff
)

__all__ = [
    # --- dataResults.py ---
    "clean_nan_values", 
    "get_results",
    # --- resume.py ---
    "load_completed_results",
    "append_result",
    "is_rate_limited",
    # --- timecho.py ---
    "forecast",
    "extract_pred_values",
    "calc_metrics",
    "calc_diff"
]

