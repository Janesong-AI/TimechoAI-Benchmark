"""
timecho.py —— TimechoAI CRUD Wrapper

Provides a general calling layer for Timecho prediction interface, including:
  - forecast(): Wraps API call, timing, and exception handling
  - extract_pred_values(): Extracts prediction values from API-returned result DataFrame
  - calc_metrics(): Calculates MAE / RMSE / MAPE
  - calc_diff(): Calculates mean absolute difference between two sets of prediction values
"""

# ============================================================
import aiohttp

_original_aiohttp_request = aiohttp.ClientSession._request

async def _hooked_aiohttp_request(self, method, url, **kwargs):
    """Intercept aiohttp requests to capture full response headers for 429"""
    resp = await _original_aiohttp_request(self, method, url, **kwargs)
    if resp.status == 429:
        print("\n" + "=" * 60)
        print("[429 Interceptor] Caught Too Many Requests (aiohttp)")
        print(f"  URL: {url}")
        print(f"  Status Code: {resp.status}")
        print(f"  Retry-After: {resp.headers.get('Retry-After', 'Not returned')}")
        print(f"  X-RateLimit-Remaining: {resp.headers.get('X-RateLimit-Remaining', 'Not returned')}")
        print(f"  X-RateLimit-Reset: {resp.headers.get('X-RateLimit-Reset', 'Not returned')}")
        print(f"  All Response Headers:")
        for k, v in resp.headers.items():
            print(f"    {k}: {v}")
        print("=" * 60 + "\n")
    return resp

aiohttp.ClientSession._request = _hooked_aiohttp_request
# ============================================================

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import requests

from utils.client import get_timecho_client

# ============================================================
# Prediction Value Extraction
# ============================================================

def extract_pred_values(pred_df: pd.DataFrame) -> np.ndarray:
    """
    Extract numeric columns from prediction result DataFrame (excluding time column).

    Args:
        pred_df: Prediction result DataFrame returned by API

    Returns:
        numpy array of float type
    """
    if "target" in pred_df.columns:
        return pred_df["target"].values.astype(float)
    non_time_cols = [c for c in pred_df.columns if c != "time"]
    return pred_df[non_time_cols[0]].values.astype(float)


# ============================================================
# Prediction Call (Core Wrapper)
# ============================================================

def forecast(
    *,
    targets: pd.DataFrame,
    history_covs: pd.DataFrame | None = None,
    future_covs: pd.DataFrame | None = None,
    model_id: str = "Holt-Winters",
    output_length: int = 64,
    time_col: str = "time",
    auto_adapt: bool = True,
    api_key: str | None = None,
) -> tuple[np.ndarray | None, float, str | None]:
    """
    Call TimechoAI prediction interface and return prediction values, elapsed time, and error message.

    Args:
        targets: Historical target values DataFrame (must contain time and target columns)
        history_covs: Historical covariates DataFrame (optional)
        future_covs: Future covariates DataFrame (optional, pass None to indicate no covariates)
        model_id: Model ID
        output_length: Prediction length
        time_col: Time column name
        auto_adapt: Whether to auto-adapt
        api_key: API key (optional, uses global configuration by default)

    Returns:
        (pred_values, elapsed_ms, error_msg)
        - pred_values: Prediction value array (None on failure)
        - elapsed_ms: Elapsed time (milliseconds)
        - error_msg: Error message (None on success)
    """
    client = get_timecho_client(api_key)
    t0 = time.perf_counter()

    try:
        # Only pass to API when covariates are not None, avoiding errors from certain models due to None values
        api_kwargs: dict = {
            "targets": targets,
            "model_id": model_id,
            "output_length": output_length,
            "time_col": time_col,
            "auto_adapt": auto_adapt,
        }
        if history_covs is not None:
            api_kwargs["history_covs"] = history_covs
        if future_covs is not None:
            api_kwargs["future_covs"] = future_covs

        result = client.forecast(**api_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        pred_values = extract_pred_values(result[0])
        return pred_values, elapsed_ms, None
    except Exception as e:
        if hasattr(e, 'response'):
            resp = e.response
            print(f"Status Code: {resp.status_code}")
            print(f"Retry-After: {resp.headers.get('Retry-After', 'Not returned')}")
        else:
            print(f"Exception Type: {type(e)}")
            # print(f"Exception Attributes: {dir(e)}")
            print(f"Exception Message: {e}")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return None, elapsed_ms, str(e)


# ============================================================
# Accuracy Metrics
# ============================================================

def calc_metrics(
    pred: np.ndarray | None,
    truth: np.ndarray,
) -> dict[str, float | None]:
    """
    Calculate MAE / RMSE / MAPE.

    Args:
        pred: Prediction values
        truth: Ground truth values

    Returns:
        {"MAE": ..., "RMSE": ..., "MAPE": ...}
    """
    if pred is None:
        return {"MAE": None, "RMSE": None, "MAPE": None}
    mae = float(np.mean(np.abs(pred - truth)))
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    mape = float(np.mean(np.abs((pred - truth) / truth)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def calc_diff(pred1: np.ndarray | None, pred2: np.ndarray | None) -> float | None:
    """
    Calculate mean absolute difference between two sets of prediction values.

    Args:
        pred1: First set of prediction values
        pred2: Second set of prediction values

    Returns:
        Mean absolute difference, returns None if either is None
    """
    if pred1 is None or pred2 is None:
        return None
    return float(np.mean(np.abs(pred1 - pred2)))
