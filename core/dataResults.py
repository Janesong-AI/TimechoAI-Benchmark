#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core/dataResults.py —— Result Data Processing
====================================
Function: Provides common utility functions for data cleaning, result querying, etc.

Author: Janesong
Create Date: 2026/07/19
"""

import numpy as np
from typing import Dict, List, Optional, Any


def clean_nan_values(obj: Any) -> Any:
    """
    Recursively clean NaN values, converting to None (JSON-compatible)
    
    Args:
        obj: Object to be processed (dict, list, or other types)
    
    Returns:
        Cleaned object, NaN/Inf replaced with None
    
    Example:
        >>> data = {"mae": np.nan, "rmse": 0.5, "tags": [np.inf, 1.0]}
        >>> clean_nan_values(data)
        {'mae': None, 'rmse': 0.5, 'tags': [None, 1.0]}
    """
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        # Handle numpy numeric types
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    return obj


def get_results(results_data: List[Dict[str, Any]], 
               model_id: str, 
               scene_prefix: str, 
               pass_name: str = "Preprocessed") -> Optional[Dict[str, Any]]:
    """
    Precisely retrieve test results for a specific model, scene, and round from result list
    
    Args:
        results_data: Result data list (records read from CSV)
        model_id: Model ID (e.g., "Timer-XL-1.0", "TimesFM-2.0")
        scene_prefix: Scene prefix (e.g., "S0", "S1", "S4")
        pass_name: Round name ("Raw" or "Preprocessed")
    
    Returns:
        Matching result dictionary, returns None if not found
    
    Example:
        >>> results = [
        ...     {"model_id": "Timer-3.5", "scene": "S0-Clean[Preprocessed]", "pass": "Preprocessed", "mae": 0.5},
        ...     {"model_id": "Timer-3.5", "scene": "S1-Missing5%[Raw]", "pass": "Raw", "mae": None}
        ... ]
        >>> get_results(results, "Timer-3.5", "S0", "Preprocessed")
        {'model_id': 'Timer-XL-1.0', 'scene': 'S0-Clean[Preprocessed]', 'pass': 'Preprocessed', 'mae': 0.5}
    """
    for r in results_data:
        if r["model_id"] == model_id and r["scene"].startswith(scene_prefix) and r["pass"] == pass_name:
            return r
    return None


